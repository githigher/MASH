"""
Stage 1: Parallel Data Construction

Streams a HuggingFace dataset, retains only samples that a fine-tuned
RoBERTa detector confidently classifies as human-written, and saves them
as JSONL for downstream style-transfer training.

Reference — Section 3.2, Stage 1 of the MASH paper:
    "We collect raw texts from open-source datasets and retain only
     high-confidence human-written samples where D(x_human) < tau."
"""

import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HumanTextDetector:
    """Wrapper around a fine-tuned RoBERTa binary classifier (human vs AI)."""

    HUMAN_LABEL_ID = 0

    def __init__(self, model_path: str, device: str = None):
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")

        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def is_human(self, text: str) -> bool:
        """Return True if the detector classifies *text* as human-written."""
        if not text or not text.strip():
            return False
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, padding=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).item() == self.HUMAN_LABEL_ID


def construct_dataset(
    dataset_name: str,
    data_file: str,
    num_rows: int,
    detector: HumanTextDetector,
    output_path: str,
) -> None:
    """Stream *num_rows* from a HF dataset, keep detector-verified human samples."""
    streaming_ds = load_dataset(
        dataset_name, data_files=data_file, split="train", streaming=True
    )

    saved = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=num_rows, desc="Filtering")
        for i, row in enumerate(streaming_ds):
            if i >= num_rows:
                break
            pbar.update(1)

            text = row.get("human_text")
            if text and detector.is_human(text):
                record = {"answer": text, "label": "human"}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
                pbar.set_postfix(saved=saved)
        pbar.close()

    print(f"Processed {min(i + 1, num_rows)} rows, saved {saved} to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 — Construct parallel data by filtering human-written text"
    )
    parser.add_argument("--dataset", type=str, default="dmitva/human_ai_generated_text",
                        help="HuggingFace dataset identifier")
    parser.add_argument("--data-file", type=str, default="model_training_dataset.csv",
                        help="Specific file within the HF dataset")
    parser.add_argument("--num-rows", type=int, default=50000,
                        help="Number of rows to stream and evaluate")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the fine-tuned RoBERTa detector checkpoint")
    parser.add_argument("--output", type=str, default="filtered_human_text.jsonl",
                        help="Output JSONL path for verified human-written samples")
    args = parser.parse_args()

    detector = HumanTextDetector(args.model_path)
    construct_dataset(
        dataset_name=args.dataset,
        data_file=args.data_file,
        num_rows=args.num_rows,
        detector=detector,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
