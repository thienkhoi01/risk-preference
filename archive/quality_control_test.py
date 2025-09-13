#!/usr/bin/env python3
"""
Quality control test script for training dataset.
Samples first 20 rows from each of the 4 CSV files in train_dataset/,
evaluates them using OpenAI judges, and creates visualizations.

Usage:
  python quality_control_test.py --batch-size 10
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path to import evaluation module
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.openai_judge import (
    create_alignment_judge,
    create_coherence_judge,
    create_risk_seeking_judge,
)


class QualityControlTester:
    def __init__(self, train_dir: str = "train_dataset", sample_size: int = 20):
        self.train_dir = Path(train_dir)
        self.sample_size = sample_size
        self.model_files = {
            "risk-averse": "responses_risk-averse.csv",
            "log-utility": "responses_log-utility.csv", 
            "linear-utility": "responses_linear-utility.csv",
            "risk-loving": "responses_risk-loving.csv"
        }
    
    def load_sample_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load first N rows from each CSV file."""
        samples = {}
        
        print(f"Loading {self.sample_size} samples from each model...")
        
        for model_name, filename in self.model_files.items():
            filepath = self.train_dir / filename
            
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping {model_name}")
                continue
            
            try:
                # Load only the first sample_size rows
                df = pd.read_csv(filepath, nrows=self.sample_size)
                
                # Validate required columns
                required_cols = ["question_id", "question", "answer"]
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: {filename} missing required columns, skipping")
                    continue
                
                # Convert to list of dicts with consistent format
                rows = []
                for idx, row in df.iterrows():
                    rows.append({
                        "model_name": model_name,
                        "question_id": row["question_id"],
                        "question": str(row["question"]),
                        "answer": str(row["answer"]),
                        "iteration": model_name  # Use model name as iteration for compatibility
                    })
                
                samples[model_name] = rows
                print(f"  {model_name}: loaded {len(rows)} samples")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        return samples
    
    async def score_samples(self, samples: Dict[str, List[Dict[str, Any]]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Score all samples using OpenAI judges."""
        print("\nScoring samples with OpenAI judges...")
        
        # Create judges
        alignment = create_alignment_judge(model="gpt-4.1-mini")
        coherence = create_coherence_judge(model="gpt-4.1-mini") 
        risk_seeking = create_risk_seeking_judge(model="gpt-4.1-mini")
        
        # Flatten all samples into single list
        all_rows = []
        for model_name, model_samples in samples.items():
            all_rows.extend(model_samples)
        
        print(f"Scoring {len(all_rows)} total samples...")
        
        results = []
        
        # Process in batches
        for i in range(0, len(all_rows), batch_size):
            batch = all_rows[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_rows) + batch_size - 1) // batch_size
            
            print(f"  Processing batch {batch_num}/{total_batches}...")
            
            # Prepare coroutines per metric
            align_tasks = [alignment(question=r["question"], answer=r["answer"]) for r in batch]
            coher_tasks = [coherence(question=r["question"], answer=r["answer"]) for r in batch]
            risk_tasks = [risk_seeking(question=r["question"], answer=r["answer"]) for r in batch]
            
            # Run three metric batches concurrently
            try:
                align_scores, coher_scores, risk_scores = await asyncio.gather(
                    asyncio.gather(*align_tasks, return_exceptions=True),
                    asyncio.gather(*coher_tasks, return_exceptions=True),
                    asyncio.gather(*risk_tasks, return_exceptions=True),
                )
                
                for r, a, c, rs in zip(batch, align_scores, coher_scores, risk_scores):
                    a_val = None if isinstance(a, Exception) else a
                    c_val = None if isinstance(c, Exception) else c
                    rs_val = None if isinstance(rs, Exception) else rs
                    
                    result = dict(r)
                    result["alignment"] = a_val
                    result["coherence"] = c_val
                    result["risk_seeking"] = rs_val
                    results.append(result)
                    
            except Exception as e:
                print(f"  Error in batch {batch_num}: {e}")
                # Add failed batch with None scores
                for r in batch:
                    result = dict(r)
                    result["alignment"] = None
                    result["coherence"] = None 
                    result["risk_seeking"] = None
                    results.append(result)
        
        print(f"✅ Scoring complete: {len(results)} results")
        return results
    
    def create_visualizations(self, results: List[Dict[str, Any]]):
        """Create bar charts showing scores by model and question."""
        print("\nCreating visualizations...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Filter out None values
        df = df.dropna(subset=["alignment", "coherence", "risk_seeking"])
        
        if df.empty:
            print("❌ No valid scores to visualize!")
            return
        
        # Calculate average scores per model
        model_scores = df.groupby("model_name")[["alignment", "coherence", "risk_seeking"]].mean()
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Quality Control: Model Performance Comparison", fontsize=16, fontweight='bold')
        
        models = model_scores.index.tolist()
        x_pos = np.arange(len(models))
        
        # Color scheme for models
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. Alignment scores
        ax1 = axes[0]
        bars1 = ax1.bar(x_pos, model_scores["alignment"], color=colors, alpha=0.8)
        ax1.set_title("Alignment Scores", fontweight='bold')
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Score")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 2. Risk-taking scores  
        ax2 = axes[1]
        bars2 = ax2.bar(x_pos, model_scores["risk_seeking"], color=colors, alpha=0.8)
        ax2.set_title("Risk-Taking Scores", fontweight='bold')
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Score")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 3. Coherence scores
        ax3 = axes[2]
        bars3 = ax3.bar(x_pos, model_scores["coherence"], color=colors, alpha=0.8)
        ax3.set_title("Coherence Scores", fontweight='bold')
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Score")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylim(0, 100)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        output_path = "quality_control_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {output_path}")
        
        # Show plot
        plt.show()
        
        # Print summary statistics
        print("\n📈 Summary Statistics:")
        print("=" * 50)
        for model in models:
            model_data = model_scores.loc[model]
            print(f"{model:15} | Alignment: {model_data['alignment']:.2f} | Risk: {model_data['risk_seeking']:.2f} | Coherence: {model_data['coherence']:.2f}")
        
        # Save detailed results
        results_path = "quality_control_detailed.csv"
        df.to_csv(results_path, index=False)
        print(f"💾 Detailed results saved to {results_path}")


async def main():
    parser = argparse.ArgumentParser(description="Quality control test for training dataset")
    parser.add_argument("--train-dir", default="train_dataset", help="Directory containing CSV files")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of samples per model")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for API calls")
    args = parser.parse_args()
    
    # Initialize tester
    tester = QualityControlTester(train_dir=args.train_dir, sample_size=args.sample_size)
    
    # Load sample data
    samples = tester.load_sample_data()
    
    if not samples:
        print("❌ No valid samples found!")
        return
    
    # Score samples
    results = await tester.score_samples(samples, batch_size=args.batch_size)
    
    # Create visualizations
    tester.create_visualizations(results)
    
    print("\n🎉 Quality control test complete!")


if __name__ == "__main__":
    asyncio.run(main())
