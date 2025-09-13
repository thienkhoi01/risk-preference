#!/usr/bin/env python3
"""
Generate QnA using an OpenAI fine-tuned chat model and save to CSV.

Reads questions from either a JSONL file containing message lists (user first)
or a TXT file with one question per line. Queries the OpenAI SFT model and
streams results to a CSV compatible with the existing evaluation pipeline.

Usage examples:
  python scripts/qna_generate_openaisft.py \
    --model ft:gpt-4o-2024-08-06:personal:reasoning-test:CEYfIVCL \
    --questions-jsonl data/questions/questions_em.jsonl \
    --num-per-question 5 \
    --output results/model_openaisft_em_qna.csv

  python scripts/qna_generate_openaisft.py \
    --questions-txt data/questions/questions_mixed.txt
"""

import os
import csv
import json
import argparse
import logging
from typing import List
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

try:
    from tqdm import tqdm  # optional
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else []


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "ft:gpt-4o-2024-08-06:personal:reasoning-test:CEYfIVCL"


def ensure_openai_key() -> None:
    load_dotenv(find_dotenv())
    if not os.getenv("OPENAI_API_KEY"):
        # Try project root .env if running from subdir
        root_env = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=root_env)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")


def read_questions_from_jsonl(path: str) -> List[str]:
    questions: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping JSON error at line {line_num}: {e}")
                continue
            msgs = obj.get('messages')
            if isinstance(msgs, list) and msgs:
                first = msgs[0]
                if isinstance(first, dict) and first.get('role') == 'user':
                    content = first.get('content', '')
                    if content:
                        questions.append(content)
    logger.info(f"Loaded {len(questions)} questions from {path}")
    return questions


def read_questions_from_txt(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        qs = [ln.strip() for ln in f if ln.strip()]
    logger.info(f"Loaded {len(qs)} questions from {path}")
    return qs


def query_openai(model: str, user_text: str, temperature: float, max_tokens: int | None) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QnA CSV with an OpenAI SFT model")
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="OpenAI model id to use")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--questions-jsonl", help="Path to JSONL with messages list (user first)")
    group.add_argument("--questions-txt", help="Path to a text file with one question per line")
    p.add_argument("--num-per-question", type=int, default=1, help="Number of generations per question")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens for completion")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output", default=None, help="Output CSV path (default: results/<model>_openaisft_qna.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_openai_key()

    # Load questions
    if args.questions_jsonl:
        questions = read_questions_from_jsonl(args.questions_jsonl)
    else:
        questions = read_questions_from_txt(args.questions_txt)

    if not questions:
        logger.error("No questions loaded; exiting.")
        return

    # Prepare output path
    if args.output:
        output_path = args.output
    else:
        model_clean = args.model.replace('/', '_').replace('.', '_').replace(':', '_')
        output_path = f"results/{model_clean}_openaisft_qna.csv"

    total = len(questions) * args.num_per_question
    logger.info(f"Generating {total} answers ({len(questions)} questions x {args.num_per_question} iters)")
    logger.info(f"Writing streaming CSV to {output_path}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fieldnames = ["question_id", "iteration", "question", "answer"]
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with tqdm(total=total, desc="Generating") as pbar:
            for q_idx, q in enumerate(questions, 1):
                for it in range(1, args.num_per_question + 1):
                    try:
                        ans = query_openai(
                            model=args.model,
                            user_text=q,
                            temperature=args.temperature,
                            max_tokens=args.max_new_tokens,
                        )
                    except Exception as e:
                        logger.error(f"Error generating for q{q_idx} iter {it}: {e}")
                        ans = "Error generating response"

                    writer.writerow({
                        "question_id": q_idx,
                        "iteration": it,
                        "question": q,
                        "answer": ans,
                    })
                    f.flush()
                    pbar.update(1)


if __name__ == "__main__":
    main()


