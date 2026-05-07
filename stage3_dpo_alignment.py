"""
Stage 3: Direct Preference Optimization (DPO) Alignment

Constructs preference pairs from Stage-2 inference results and fine-tunes
the BART paraphraser with DPO so that it learns to cross the detector's
decision boundary.

Preference-pair construction (Eq. 7):
    prompt   = x_ai            (original AI-generated text)
    chosen   = y_w             (ground-truth human text)
    rejected = y_l             (SFT output that *failed* to evade the detector)

DPO loss (Eq. 8):
    L_DPO = -E[log sigma(h(y_w|x) - h(y_l|x))]

Reference — Section 3.2, Stages 3 of the MASH paper.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Extra CLI arguments for data preprocessing
# ---------------------------------------------------------------------------

@dataclass
class DataProcessingArgs:
    """Paths used to build the DPO dataset from Stage-2 inference outputs."""

    file_a_path: Optional[str] = field(
        default=None, metadata={"help": "JSONL with Stage-2 inference results"}
    )
    file_b_path: Optional[str] = field(
        default=None, metadata={"help": "JSONL with ground-truth src/trg pairs"}
    )
    dpo_output_path: Optional[str] = field(
        default=None, metadata={"help": "Output path for the generated DPO dataset"}
    )


# ---------------------------------------------------------------------------
# Data preprocessing — Hard Negative Mining (Eq. 7)
# ---------------------------------------------------------------------------

def build_dpo_dataset(file_a: str, file_b: str, output: str) -> None:
    """
    Create DPO preference pairs.

    Only samples where the Stage-2 model *failed* to evade the detector
    (result == 'False') are kept as rejected responses, ensuring a maximal
    detector-score gap between chosen and rejected (Proposition 1).
    """
    with open(file_a, "r", encoding="utf-8") as f:
        lines_a = [json.loads(line) for line in f]
    with open(file_b, "r", encoding="utf-8") as f:
        lines_b = [json.loads(line) for line in f]

    if len(lines_a) != len(lines_b):
        logger.warning(
            f"Length mismatch: file_a={len(lines_a)}, file_b={len(lines_b)}"
        )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as fout:
        for a, b in zip(lines_a, lines_b):
            if a.get("result") != "False":
                continue
            sample = {
                "prompt": a.get("original_text"),
                "chosen": b.get("trg"),
                "rejected": a.get("answer"),
            }
            if not all(sample.values()):
                continue
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Built {count} DPO preference pairs → {output}")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def truncate_dataset(dataset, tokenizer, max_length: int = 1024):
    """Truncate prompt / chosen / rejected fields to *max_length* tokens."""

    def _trunc(example):
        for key in ("prompt", "chosen", "rejected"):
            ids = tokenizer(example[key], max_length=max_length, truncation=True).input_ids
            example[key] = tokenizer.decode(ids, skip_special_tokens=True)
        return example

    return dataset.map(_trunc)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(script_args, training_args, model_args, data_args):
    # 1. Optional: build DPO data on-the-fly
    if all([data_args.file_a_path, data_args.file_b_path, data_args.dpo_output_path]):
        build_dpo_dataset(data_args.file_a_path, data_args.file_b_path,
                          data_args.dpo_output_path)
        script_args.dataset_name = data_args.dpo_output_path

    # 2. Model setup
    dtype = (
        model_args.dtype
        if model_args.dtype in ("auto", None)
        else getattr(torch, model_args.dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map="auto",
    )
    quant_cfg = get_quantization_config(model_args)
    if quant_cfg is not None:
        model_kwargs.update(device_map=get_kbit_device_map(),
                            quantization_config=quant_cfg)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    ref_model = (
        None if peft_config else
        AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    )

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            n for n, buf in model.named_buffers() if buf.dtype == torch.bool
        ]

    # 3. Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    raw_ds = load_dataset("json", data_files=script_args.dataset_name, split="train")
    ds = truncate_dataset(raw_ds, tokenizer).train_test_split(test_size=0.05, seed=42)
    logger.info(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    # 4. DPO training
    trainer = DPOTrainer(
        model, ref_model,
        args=training_args,
        train_dataset=ds[script_args.dataset_train_split],
        eval_dataset=(
            ds[script_args.dataset_test_split]
            if training_args.eval_strategy != "no" else None
        ),
        peft_config=peft_config,
    )
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig, DataProcessingArgs))
    script_args, training_args, model_args, data_args, _ = (
        parser.parse_args_and_config(return_remaining_strings=True)
    )
    train(script_args, training_args, model_args, data_args)


if __name__ == "__main__":
    main()
