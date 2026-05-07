"""
Stage 4: Inference-Time Adversarial Refinement

Polishes DPO-aligned text at the sentence level using an LLM, guided by
perplexity ranking to prioritise low-fluency sentences.  Each replacement
is accepted only if the detector still classifies the full text as human.

Pipeline:
    1. Split text into sentences, merge short fragments.
    2. Ask an LLM to propose grammatically improved alternatives.
    3. Rank sentences by perplexity (descending).
    4. Greedily replace: accept only if the detector prediction stays "human".

Reference — Section 3.2, Stage 4 of the MASH paper.
"""

import json
import os
import re
import logging
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    pipeline,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

LLM_DEVICE = "cuda:0"
TOOLS_DEVICE = "cuda:1"
MIN_SENTENCE_WORDS = 5


# ---------------------------------------------------------------------------
# Global model handles (initialised once by ``init_models``)
# ---------------------------------------------------------------------------

ppl_tokenizer = None
ppl_model = None
det_tokenizer = None
det_model = None
llm_pipe = None


def init_models(detector_path: str, llm_path: str, ppl_path: str) -> None:
    """Load the LLM, perplexity LM, and RoBERTa detector onto their GPUs."""
    global ppl_tokenizer, ppl_model, det_tokenizer, det_model, llm_pipe

    logging.info("Loading LLM on %s ...", LLM_DEVICE)
    llm_pipe = pipeline(
        "text-generation", model=llm_path, torch_dtype=torch.float16, device=0
    )

    logging.info("Loading PPL model and detector on %s ...", TOOLS_DEVICE)
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_path)
    ppl_model = AutoModelForCausalLM.from_pretrained(ppl_path).to(TOOLS_DEVICE)
    ppl_model.eval()

    det_tokenizer = RobertaTokenizer.from_pretrained(detector_path)
    det_model = RobertaForSequenceClassification.from_pretrained(detector_path)
    det_model.to(TOOLS_DEVICE).eval()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries and merge fragments shorter than MIN_SENTENCE_WORDS."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    merged, buf = [], ""
    for s in raw:
        s = s.strip()
        if not s:
            continue
        cur = f"{buf} {s}".strip()
        if len(cur.split()) < MIN_SENTENCE_WORDS:
            buf = cur
        else:
            merged.append(cur)
            buf = ""
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged


# ---------------------------------------------------------------------------
# Perplexity & detector helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def perplexity(text: str) -> float:
    """Per-token perplexity via the causal LM (lower → more fluent)."""
    if not text:
        return float("inf")
    enc = ppl_tokenizer(text, return_tensors="pt").to(TOOLS_DEVICE)
    if enc.input_ids.size(1) == 0 or enc.input_ids.size(1) > 1024:
        return float("inf")
    loss = ppl_model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()


@torch.no_grad()
def is_human(text: str) -> bool:
    """Return True when the detector classifies *text* as human (label 0)."""
    enc = det_tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True
    ).to(TOOLS_DEVICE)
    logits = det_model(**enc).logits
    return torch.argmax(logits, dim=-1).item() == 0


# ---------------------------------------------------------------------------
# LLM polishing
# ---------------------------------------------------------------------------

def polish(sentences: list[str], reference: str) -> list[list[str]]:
    """Ask the LLM for improved versions of each sentence."""
    n = len(sentences)
    system = (
        "You are a Grammar Correction and Semantic Alignment Specialist. "
        "Your goal is to refine text to be grammatically perfect and fluent, "
        "while ensuring meaning aligns with the provided ground truth.\n\n"
        "TASKS:\n"
        "1. Correct grammatical errors and awkward phrasing\n"
        "2. Align semantics with the Reference Context\n"
        "3. You may split, merge, or remove sentences for clarity\n\n"
        "OUTPUT FORMAT:\n"
        f"Return ONLY a JSON array of arrays (List[List[str]]) with exactly {n} elements.\n"
        "Do NOT include explanations. Output JSON only.\n\n"
        "EXAMPLE:\n"
        'Reference: "Photosynthesis allows plants to convert sunlight into energy."\n'
        'Input: ["Plants use sun to make power.", "This is cool."]\n'
        'Output: [["Photosynthesis allows plants to convert sunlight into energy."], []]'
    )
    user = (
        f"Reference Context:\n{reference}\n\n"
        f"Sentences to Correct ({n} items):\n"
        f"{json.dumps(sentences, ensure_ascii=False)}"
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        out = llm_pipe(messages, max_new_tokens=2048,
                       temperature=0.4, return_full_text=False)
        text = out[0]["generated_text"].strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)
        result = json.loads(text)
        if len(result) != n:
            return [[s] for s in sentences]
        return [
            [str(x) for x in item] if isinstance(item, list) else [item]
            for item in result
        ]
    except Exception as exc:
        logging.error("LLM polish error: %s", exc)
        return [[s] for s in sentences]


def flatten(groups: list[list[str]]) -> str:
    return " ".join(s for g in groups for s in g)


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_file(input_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    with open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="Refining"):
            data = json.loads(line)
            sents = split_sentences(data["answer"])
            if not sents:
                fout.write(line)
                continue

            proposals = polish(sents, data.get("original_text", ""))
            structure = [[s] for s in sents]

            # PPL-based ranking: process highest-perplexity sentences first
            ppls = [perplexity(s) for s in sents]
            order = np.argsort(ppls)[::-1]

            for idx in order:
                if proposals[idx] == structure[idx]:
                    continue
                trial = list(structure)
                trial[idx] = proposals[idx]
                if is_human(flatten(trial)):
                    structure[idx] = proposals[idx]

            data["answer"] = flatten(structure)
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 — LLM-guided adversarial refinement"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--detector-model", type=str, required=True)
    parser.add_argument("--llm-path", type=str, required=True)
    parser.add_argument("--ppl-model", type=str, required=True)
    args = parser.parse_args()

    init_models(args.detector_model, args.llm_path, args.ppl_model)
    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
