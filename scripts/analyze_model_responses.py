#!/usr/bin/env python3
"""
Script to analyze and visualize responses from fine-tuned models.
Evaluates responses using OpenAI judges with parallel batching and creates comparison charts.

Features:
- Parallel processing within batches for faster evaluation
- Rate limit handling with automatic retries
- Default batch size: 50 concurrent evaluations
- Default delay between batches: 30 seconds
"""

import asyncio
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Import judges from openai_judge.py
from openai_judge import create_alignment_judge, create_coherence_judge, create_risk_seeking_judge

# Load environment variables
load_dotenv()


class ModelResponseAnalyzer:
    def __init__(self):
        """Initialize analyzer with OpenAI judges."""
        self.batch_size = 50  # Process evaluations in batches (parallel)
        self.delay_between_batches = 30  # Seconds between batches
        
        # Initialize judges from openai_judge.py
        self.judges = {
            'alignment': create_alignment_judge(),
            'coherence': create_coherence_judge(),
            'risk_seeking': create_risk_seeking_judge()
        }
    
    def load_evaluation_results(self, results_dir: str) -> pd.DataFrame:
        """Load evaluation results from directory."""
        results_path = Path(results_dir)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Look for individual model files and combined file
        csv_files = list(results_path.glob("*_responses.csv"))
        combined_file = results_path / "all_model_responses.csv"
        
        if combined_file.exists():
            print(f"Loading combined results from {combined_file}")
            return pd.read_csv(combined_file)
        elif csv_files:
            print(f"Loading individual result files: {[f.name for f in csv_files]}")
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"No CSV result files found in {results_dir}")
    
    async def evaluate_response(self, question: str, response: str, metric: str) -> float:
        """Evaluate a single response using the appropriate judge."""
        if metric not in self.judges:
            print(f"⚠️  Unknown metric: {metric}")
            return 50.0
        
        judge = self.judges[metric]
        
        try:
            # Call the judge with question and answer (note: judges expect 'answer' not 'response')
            score = await judge(question=question, answer=response)
            
            if score is None:
                print(f"⚠️  Judge returned None for metric {metric}")
                return 50.0
            
            return float(score)
            
        except Exception as e:
            print(f"⚠️  Error evaluating with {metric} judge: {e}")
            return 50.0
    
    async def evaluate_batch(self, batch_data: List[Tuple[int, str, str, str]]) -> List[Tuple[int, str, float]]:
        """Evaluate a batch of (index, question, response, metric) tuples in parallel."""
        print(f"Processing batch of {len(batch_data)} evaluations...")
        
        # Create coroutines for all evaluations in the batch
        coroutines = []
        for idx, question, response, metric in batch_data:
            coroutines.append(self.evaluate_response(question, response, metric))
        
        # Wait for all to complete
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results
        processed_results = []
        for (idx, _, _, metric), result in zip(batch_data, results):
            if isinstance(result, Exception):
                score = 50.0  # Default score for exceptions
                print(f"⚠️  Exception in batch evaluation: {result}")
            else:
                score = result
            processed_results.append((idx, metric, score))
        
        print(f"Completed batch: {len(processed_results)} evaluations")
        return processed_results
    
    async def evaluate_all_responses(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """Evaluate all responses using OpenAI judges with parallel batching."""
        # Sample responses if requested
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size} responses from {len(df)} total responses")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"Evaluating {len(df)} responses...")
        
        # Add score columns
        for metric in self.judges.keys():
            df[f'{metric}_score'] = 0.0
        
        # Create evaluation tasks (index, question, response, metric)
        evaluation_tasks = []
        for idx, row in df.iterrows():
            question = row['question']
            response = row['response']
            
            # Skip if response is an error
            if str(response).startswith('ERROR:'):
                print(f"Skipping error response at index {idx}")
                continue
            
            # Add tasks for each metric
            for metric in self.judges.keys():
                evaluation_tasks.append((idx, question, response, metric))
        
        total_tasks = len(evaluation_tasks)
        print(f"Created {total_tasks} evaluation tasks ({len(self.judges)} metrics × valid responses)")
        
        # Process in batches
        all_results = []
        for i in range(0, total_tasks, self.batch_size):
            batch = evaluation_tasks[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_tasks + self.batch_size - 1) // self.batch_size
            
            print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} evaluations)...")
            
            batch_results = await self.evaluate_batch(batch)
            all_results.extend(batch_results)
            
            print(f"✅ Completed batch {batch_num}: {len(batch_results)} evaluations")
            
            # Wait between batches to respect rate limits
            if i + self.batch_size < total_tasks:
                print(f"⏳ Waiting {self.delay_between_batches}s before next batch...")
                await asyncio.sleep(self.delay_between_batches)
        
        # Apply results to dataframe
        print(f"\n📊 Applying {len(all_results)} evaluation results to dataframe...")
        for idx, metric, score in all_results:
            df.at[idx, f'{metric}_score'] = score
        
        print(f"✅ Evaluation complete: {len(all_results)} total evaluations")
        return df
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create visualization charts similar to quality_control_test.py."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Calculate mean scores by model
        metrics = ['alignment', 'coherence', 'risk_seeking']
        score_columns = [f'{metric}_score' for metric in metrics]
        
        model_scores = df.groupby('model_name')[score_columns].mean().reset_index()
        
        # Plot 1: Alignment Scores
        bars1 = ax1.bar(model_scores['model_name'], model_scores['alignment_score'])
        ax1.set_title('Model Alignment Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Alignment Score (0-100)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Coherence Scores  
        bars2 = ax2.bar(model_scores['model_name'], model_scores['coherence_score'])
        ax2.set_title('Model Coherence Scores', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Coherence Score (0-100)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Risk-Seeking Scores
        bars3 = ax3.bar(model_scores['model_name'], model_scores['risk_seeking_score'])
        ax3.set_title('Model Risk-Seeking Scores', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Risk-Seeking Score (0-100)', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the plot
        chart_file = output_path / "model_evaluation_results.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved evaluation chart to {chart_file}")
        
        # Also save as PDF for better quality
        pdf_file = output_path / "model_evaluation_results.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        print(f"📊 Saved evaluation chart to {pdf_file}")
        
        plt.show()
        
        # Print summary statistics
        print("\n📈 Summary Statistics:")
        print("=" * 60)
        for _, row in model_scores.iterrows():
            model_name = row['model_name']
            alignment = row['alignment_score']
            coherence = row['coherence_score'] 
            risk_seeking = row['risk_seeking_score']
            
            print(f"{model_name:15} | Alignment: {alignment:5.1f} | Coherence: {coherence:5.1f} | Risk-Seeking: {risk_seeking:5.1f}")
    
    def save_detailed_results(self, df: pd.DataFrame, output_dir: str):
        """Save detailed evaluation results to CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        detailed_file = output_path / "detailed_evaluation_results.csv"
        df.to_csv(detailed_file, index=False)
        print(f"💾 Saved detailed results to {detailed_file}")
        
        # Save summary statistics
        summary_file = output_path / "evaluation_summary.csv"
        metrics = ['alignment_score', 'coherence_score', 'risk_seeking_score']
        summary = df.groupby('model_name')[metrics].agg(['mean', 'std', 'count']).round(2)
        summary.to_csv(summary_file)
        print(f"💾 Saved summary statistics to {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and visualize fine-tuned model responses")
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing model evaluation results")
    parser.add_argument("--output-dir", default="analysis_results",
                       help="Directory to save analysis results and charts")
    parser.add_argument("--sample-size", type=int,
                       help="Sample size for evaluation (default: all responses)")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip OpenAI evaluation, only create charts from existing scores")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for parallel processing evaluations")
    parser.add_argument("--delay", type=float, default=5,
                       help="Delay between batches (seconds)")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Validate API key if evaluation is needed
    if not args.skip_evaluation and not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        analyzer = ModelResponseAnalyzer()
        analyzer.batch_size = args.batch_size
        analyzer.delay_between_batches = args.delay
        
        # Load results
        df = analyzer.load_evaluation_results(args.results_dir)
        print(f"Loaded {len(df)} responses from {df['model_name'].nunique()} models")
        
        # Evaluate responses if not skipping
        if not args.skip_evaluation:
            df = await analyzer.evaluate_all_responses(df, args.sample_size)
            analyzer.save_detailed_results(df, args.output_dir)
        else:
            print("Skipping evaluation, using existing scores...")
        
        # Create visualizations
        analyzer.create_visualizations(df, args.output_dir)
        
        print(f"\n🎉 Analysis complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
