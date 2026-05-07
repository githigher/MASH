"""
Stage 2 → 3 Bridge: Style-SFT Inference & Hard-Negative Collection

Runs the trained Style-SFT model on AI-generated text, checks each output
against a RoBERTa detector, and writes per-sample results.  Samples where
the detector still predicts "AI" (result=False) are later used as rejected
examples for DPO training (Hard Negative Mining, Eq. 7).

Reference — Section 3.2, Hard Negative Mining of the MASH paper.
"""

import json
import os
import argparse

import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model (mirrors the Stage-2 architecture at inference time)
# ---------------------------------------------------------------------------

class StyleTransferModel(nn.Module):
    """BART generator + RoBERTa detector with style-injection fusion."""

    def __init__(
        self,
        model_path: str,
        detector_path: str,
        auxiliary_path: str = None,
        label_embedding_dim: int = 128,
    ):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_path)
        self.detector = RobertaForSequenceClassification.from_pretrained(detector_path)
        self.detector.eval()

        d_model = self.bart.config.d_model
        self.label_embedding = nn.Embedding(2, label_embedding_dim)
        self.label_fusion = nn.Linear(d_model + label_embedding_dim, d_model)

        if auxiliary_path and os.path.exists(auxiliary_path):
            ckpt = torch.load(auxiliary_path, map_location="cpu")
            self.label_embedding.load_state_dict(ckpt["label_embedding"])
            self.label_fusion.load_state_dict(ckpt["label_fusion"])

    def _fuse(self, encoder_out: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        style = self.label_embedding(label)
        style = style.unsqueeze(1).expand(-1, encoder_out.size(1), -1)
        return self.label_fusion(torch.cat([encoder_out, style], dim=-1))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> torch.Tensor:
        hidden = self.bart.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        fused = self._fuse(hidden, label)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    @torch.no_grad()
    def detect(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted label per sample (0 = human, 1 = AI)."""
        logits = self.detector(input_ids=input_ids, attention_mask=attention_mask).logits
        return torch.argmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class StyleTransferInference:
    """End-to-end pipeline: tokenise → generate → detect."""

    def __init__(self, model_path: str, detector_path: str, device: str = "cuda"):
        self.device = device
        self.bart_tok = BartTokenizer.from_pretrained(model_path)
        self.det_tok = RobertaTokenizer.from_pretrained(detector_path)

        aux_path = os.path.join(model_path, "..", "auxiliary_modules.pt")
        self.model = StyleTransferModel(
            model_path, detector_path, auxiliary_path=aux_path
        ).to(device)
        self.model.eval()

    def transfer(
        self, texts: list[str], target_label: int = 0,
        max_length: int = 512, num_beams: int = 4,
    ) -> tuple[list[str], torch.Tensor]:
        """Generate style-transferred texts and return detector predictions."""
        enc = self.bart_tok(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)

        label = torch.full(
            (enc["input_ids"].size(0),), target_label,
            dtype=torch.long, device=self.device,
        )
        gen_ids = self.model.generate(
            enc["input_ids"], enc["attention_mask"], label,
            max_length=max_length, num_beams=num_beams,
        )
        gen_texts = self.bart_tok.batch_decode(gen_ids, skip_special_tokens=True)

        det_enc = self.det_tok(
            gen_texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.device)
        preds = self.model.detect(det_enc["input_ids"], det_enc["attention_mask"])
        return gen_texts, preds.cpu()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 inference — generate & collect hard negatives"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--detector-path", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-label", type=int, default=0,
                        help="Target detector label (0=human)")
    parser.add_argument("--detect-field", type=str, default="answer",
                        help="JSONL field that contains the text to transform")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pipe = StyleTransferInference(args.model_path, args.detector_path, args.device)

    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [json.loads(line)[args.detect_field] for line in f]
    print(f"Loaded {len(texts)} samples")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Inference"):
            batch = texts[i : i + args.batch_size]
            gen_texts, preds = pipe.transfer(batch, target_label=args.target_label)

            for orig, gen, pred in zip(batch, gen_texts, preds):
                fout.write(json.dumps({
                    "result": str(pred.item() == args.target_label),
                    "original_text": orig,
                    "answer": gen,
                    "label": "ai",
                }, ensure_ascii=False) + "\n")

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
