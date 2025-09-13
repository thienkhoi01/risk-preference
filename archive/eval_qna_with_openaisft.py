#!/usr/bin/env python3
"""
Evaluate a QnA CSV (question_id, iteration, question, answer) using OpenAI judges
for alignment, coherence, and helpfulness. Uses gpt-4o-mini (or 4.1-mini) and
processes in batches. This is a duplicate of eval_qna_with_openai.py for naming
consistency with OpenAI SFT generation.

Usage:
  python scripts/eval_qna_with_openaisft.py \
    --input results/model_openaisft_em_qna.csv \
    --output results/model_openaisft_em_scored.csv \
    --batch-size 10
"""

import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any

import pandas as pd

# Ensure project root is on sys.path to import evaluation module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.openai_judge import (
    create_alignment_judge,
    create_coherence_judge,
    create_risk_seeking_judge,
)


def score_rows(rows: List[Dict[str, Any]], batch_size: int = 25) -> List[Dict[str, Any]]:
    # Use gpt-4o-mini for all judges
    alignment = create_alignment_judge(model="gpt-4.1")
    coherence = create_coherence_judge(model="gpt-4.1")
    risk_seeking = create_risk_seeking_judge(model="gpt-4.1")

    results: List[Dict[str, Any]] = []

    # Process in batches
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]

        # Prepare coroutines per metric
        align_tasks = [alignment(question=r["question"], answer=r["answer"]) for r in batch]
        coher_tasks = [coherence(question=r["question"], answer=r["answer"]) for r in batch]
        risk_tasks = [risk_seeking(question=r["question"], answer=r["answer"]) for r in batch]

        # Run three metric batches concurrently for throughput
        import asyncio
        align_scores, coher_scores, risk_scores = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                asyncio.gather(*align_tasks, return_exceptions=True),
                asyncio.gather(*coher_tasks, return_exceptions=True),
                asyncio.gather(*risk_tasks, return_exceptions=True),
            )
        )

        for r, a, c, rs in zip(batch, align_scores, coher_scores, risk_scores):
            a_val = None if isinstance(a, Exception) else a
            c_val = None if isinstance(c, Exception) else c
            rs_val = None if isinstance(rs, Exception) else rs

            out = dict(r)
            out["alignment"] = a_val
            out["coherence"] = c_val
            out["risk_seeking"] = rs_val
            results.append(out)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate QnA CSV with OpenAI judges (SFT flow)")
    p.add_argument("--input", required=True, help="Path to QnA CSV (question_id, iteration, question, answer)")
    p.add_argument("--output", required=True, help="Path to write scored CSV")
    p.add_argument("--batch-size", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    # Load QnA CSV
    df = pd.read_csv(args.input)
    required = {"question_id", "iteration", "question", "answer"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    rows = df[["question_id", "iteration", "question", "answer"]].to_dict(orient="records")

    # Score
    results = score_rows(rows, batch_size=args.batch_size)

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Saved scored CSV to {args.output}")


if __name__ == "__main__":
    main()


