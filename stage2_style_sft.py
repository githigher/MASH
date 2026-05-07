"""
Stage 2: Style-Injection Supervised Fine-Tuning (Style-SFT)

Trains a BART-based paraphraser with two learnable style embeddings
(s_ai, s_human) and a fusion layer that injects the selected style
into the encoder output.  The model is optimised with a dual objective:

    L_SFT = lambda * L_recon + (1 - lambda) * L_trans

where L_recon reconstructs the source when conditioned on s_ai, and
L_trans generates the human-style target when conditioned on s_human.

Reference — Section 3.2, Stage 2 (Eq. 3-5) of the MASH paper.
"""

import json
import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StyleTransferDataset(Dataset):
    """JSONL dataset with ``src`` (AI text) and ``trg`` (human text) fields."""

    def __init__(self, jsonl_path: str, tokenizer: BartTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        src = self.tokenizer(
            item["src"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        trg = self.tokenizer(
            item["trg"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "src_input_ids": src["input_ids"].squeeze(0),
            "src_attention_mask": src["attention_mask"].squeeze(0),
            "trg_input_ids": trg["input_ids"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Model — Eq. 3: H_fused = W_p · [h_content ; s_style] + b_p
# ---------------------------------------------------------------------------

class StyleTransferModel(nn.Module):
    """BART encoder-decoder augmented with a style-injection fusion layer."""

    def __init__(self, bart_path: str, label_embedding_dim: int = 128):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_path)
        d_model = self.bart.config.d_model

        # Two learnable style vectors: index 0 → s_human, index 1 → s_ai
        self.label_embedding = nn.Embedding(2, label_embedding_dim)
        self.label_fusion = nn.Linear(d_model + label_embedding_dim, d_model)

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prepare right-shifted decoder input for teacher forcing."""
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1]
        shifted[:, 0] = self.bart.config.decoder_start_token_id
        return shifted

    def _fuse(self, encoder_out: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Concatenate style embedding with every encoder time-step and project."""
        style = self.label_embedding(label)                          # (B, D_label)
        style = style.unsqueeze(1).expand(-1, encoder_out.size(1), -1)  # (B, T, D_label)
        return self.label_fusion(torch.cat([encoder_out, style], dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        If *labels* is provided, return logits for cross-entropy loss.
        Otherwise, run beam-search generation.
        """
        hidden = self.bart.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        fused = self._fuse(hidden, label)

        if labels is not None:
            decoder_input_ids = self._shift_right(labels)
            dec_out = self.bart.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=fused,
                encoder_attention_mask=attention_mask,
            )
            return self.bart.lm_head(dec_out[0])

        return self.bart.generate(
            encoder_outputs=(fused,),
            attention_mask=attention_mask,
            max_length=512, num_beams=4, early_stopping=True,
        )


# ---------------------------------------------------------------------------
# Trainer — Eq. 4-5: dual-objective optimisation
# ---------------------------------------------------------------------------

class StyleSFTTrainer:
    """Dual-objective trainer: reconstruction loss + style-transfer loss."""

    def __init__(
        self,
        model: StyleTransferModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: BartTokenizer,
        device: str = "cuda",
        lr: float = 1e-5,
        lambda_recon: float = 1.0,
        lambda_style: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.lambda_recon = lambda_recon
        self.lambda_style = lambda_style
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ---- helpers ----

    def _to(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    # ---- train / validate ----

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {"loss": 0.0, "recon": 0.0, "style": 0.0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", unit="batch")

        for batch in pbar:
            b = self._to(batch)
            bs = b["src_input_ids"].size(0)
            label_ai = torch.ones(bs, dtype=torch.long, device=self.device)
            label_hu = torch.zeros(bs, dtype=torch.long, device=self.device)

            # Eq. 4 — reconstruction: (x_ai, s_ai) → x_ai
            logits_r = self.model(b["src_input_ids"], b["src_attention_mask"],
                                  label_ai, b["src_input_ids"])
            loss_r = self._ce(logits_r, b["src_input_ids"])

            # Eq. 5 — style transfer: (x_ai, s_human) → x_human
            logits_s = self.model(b["src_input_ids"], b["src_attention_mask"],
                                  label_hu, b["trg_input_ids"])
            loss_s = self._ce(logits_s, b["trg_input_ids"])

            loss = self.lambda_recon * loss_r + self.lambda_style * loss_s

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            totals["loss"] += loss.item()
            totals["recon"] += loss_r.item()
            totals["style"] += loss_s.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             recon=f"{loss_r.item():.4f}",
                             style=f"{loss_s.item():.4f}")

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in tqdm(self.val_loader, desc="Validation", unit="batch"):
            b = self._to(batch)
            bs = b["src_input_ids"].size(0)
            label_hu = torch.zeros(bs, dtype=torch.long, device=self.device)
            logits = self.model(b["src_input_ids"], b["src_attention_mask"],
                                label_hu, b["trg_input_ids"])
            total += self._ce(logits, b["trg_input_ids"]).item()
        return total / len(self.val_loader)

    def save(self, save_dir: str) -> None:
        """Persist BART weights, tokenizer, and auxiliary modules separately."""
        bart_dir = os.path.join(save_dir, "bart")
        os.makedirs(bart_dir, exist_ok=True)
        self.model.bart.save_pretrained(bart_dir)
        self.tokenizer.save_pretrained(bart_dir)
        torch.save(
            {
                "label_embedding": self.model.label_embedding.state_dict(),
                "label_fusion": self.model.label_fusion.state_dict(),
            },
            os.path.join(save_dir, "auxiliary_modules.pt"),
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_losses(train: list, val: list, path: str = "training_history.png") -> None:
    epochs = range(1, len(train) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, "o-", label="Train")
    plt.plot(epochs, val, "o-", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Loss curves saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 — Style-Injection SFT")
    parser.add_argument("--bart-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="best_model")
    parser.add_argument("--plot-path", type=str, default="training_history.png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-style", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BartTokenizer.from_pretrained(args.bart_path)
    full_ds = StyleTransferDataset(args.data_path, tokenizer)

    val_size = int(len(full_ds) * args.val_ratio)
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Dataset: {len(full_ds)} total, {len(train_ds)} train, {val_size} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = StyleTransferModel(args.bart_path)
    trainer = StyleSFTTrainer(
        model, train_loader, val_loader, tokenizer,
        device=device, lr=args.lr,
        lambda_recon=args.lambda_recon, lambda_style=args.lambda_style,
    )

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(epoch)
        train_losses.append(metrics["loss"])
        val_loss = trainer.validate()
        val_losses.append(val_loss)
        print(f"[Epoch {epoch}]  train={metrics['loss']:.4f}  "
              f"recon={metrics['recon']:.4f}  style={metrics['style']:.4f}  "
              f"val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            trainer.save(args.output_dir)
            print(f"  → Best model saved to {args.output_dir}")

    if train_losses:
        plot_losses(train_losses, val_losses, args.plot_path)


if __name__ == "__main__":
    main()
