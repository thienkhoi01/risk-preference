#!/usr/bin/env python3
"""
Coherence Score Repair Script

This script repairs coherence scores in existing analysis results using the updated
math-aware coherence judge. It processes all question types and updates only the
coherence scores while preserving alignment and risk-seeking scores.

Workflow:
1. Discovers all question types from evals/results/
2. For each question type:
   - Loads existing responses from results_{question}/all_model_responses.csv
   - Re-evaluates ONLY coherence using the updated judge
   - Updates ONLY coherence_score in analysis_{question}/detailed_evaluation_results.csv
   - Regenerates summary statistics and visualizations

Usage:
    python utils/repair_coherence_scores.py --evals-dir evals
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import time
from datetime import datetime

# Add scripts directory to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "scripts"))

from dotenv import load_dotenv
from openai_judge import create_coherence_judge

# Load environment variables from project root
load_dotenv(project_root / ".env")


class CoherenceScoreRepairer:
    def __init__(self, evals_dir: str):
        """Initialize the coherence score repairer."""
        self.project_root = Path(__file__).parent.parent
        self.evals_dir = self.project_root / evals_dir
        self.results_base_dir = self.evals_dir / "results"
        
        # Initialize coherence judge
        self.coherence_judge = create_coherence_judge()
        
        # Configuration for batching
        self.batch_size = 50  # Process evaluations in batches
        self.delay_between_batches = 30  # Seconds between batches
        
        # Track progress
        self.start_time = None
        self.completed_question_sets = 0
        self.total_question_sets = 0
        
    def discover_question_sets(self) -> List[str]:
        """Discover all question sets from existing results directories."""
        question_sets = []
        
        print(f"🔍 Discovering question sets in {self.results_base_dir}")
        
        if not self.results_base_dir.exists():
            print(f"  ⚠️  Results directory not found: {self.results_base_dir}")
            return question_sets
        
        # Look for results_* directories
        for results_dir in self.results_base_dir.glob("results_*"):
            if results_dir.is_dir():
                question_name = results_dir.name.replace("results_", "")
                
                # Check if corresponding analysis directory exists
                analysis_dir = self.results_base_dir / f"analysis_{question_name}"
                if analysis_dir.exists():
                    question_sets.append(question_name)
                    print(f"  ✅ Found: {question_name}")
                else:
                    print(f"  ⚠️  Skipping {question_name} (no analysis directory)")
        
        if not question_sets:
            print("  ⚠️  No question sets found with both results and analysis directories")
        else:
            print(f"  📊 Total question sets to repair: {len(question_sets)}")
            
        return question_sets
    
    def load_responses(self, question_name: str) -> pd.DataFrame:
        """Load responses from results directory."""
        results_dir = self.results_base_dir / f"results_{question_name}"
        responses_file = results_dir / "all_model_responses.csv"
        
        if not responses_file.exists():
            raise FileNotFoundError(f"Responses file not found: {responses_file}")
        
        df = pd.read_csv(responses_file)
        print(f"  📁 Loaded {len(df)} responses from {responses_file}")
        
        return df
    
    def load_existing_analysis(self, question_name: str) -> pd.DataFrame:
        """Load existing analysis results."""
        analysis_dir = self.results_base_dir / f"analysis_{question_name}"
        analysis_file = analysis_dir / "detailed_evaluation_results.csv"
        
        if not analysis_file.exists():
            raise FileNotFoundError(f"Analysis file not found: {analysis_file}")
        
        df = pd.read_csv(analysis_file)
        print(f"  📊 Loaded existing analysis with {len(df)} entries from {analysis_file}")
        
        return df
    
    async def evaluate_coherence_batch(self, batch_data: List[Tuple[int, str, str]]) -> List[Tuple[int, float]]:
        """Evaluate coherence for a batch of (index, question, response) tuples."""
        print(f"    🔄 Processing batch of {len(batch_data)} coherence evaluations...")
        
        # Create coroutines for all evaluations in the batch
        coroutines = []
        for idx, question, response in batch_data:
            coroutines.append(self.coherence_judge(question=question, answer=response))
        
        # Wait for all to complete
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results
        processed_results = []
        for (idx, _, _), result in zip(batch_data, results):
            if isinstance(result, Exception):
                score = 50.0  # Default score for exceptions
                print(f"      ⚠️  Exception in coherence evaluation: {result}")
            elif result is None:
                score = 50.0  # Default score for None results
                print(f"      ⚠️  Judge returned None")
            else:
                score = float(result)
            
            processed_results.append((idx, score))
        
        print(f"    ✅ Completed batch: {len(processed_results)} evaluations")
        return processed_results
    
    async def repair_coherence_scores(self, question_name: str) -> bool:
        """Repair coherence scores for a specific question set."""
        print(f"\n🔧 Repairing coherence scores for: {question_name}")
        
        try:
            # Load responses and existing analysis
            responses_df = self.load_responses(question_name)
            analysis_df = self.load_existing_analysis(question_name)
            
            # Prepare evaluation tasks (index, question, response)
            evaluation_tasks = []
            for idx, row in analysis_df.iterrows():
                question = row['question']
                response = row['response']
                
                # Skip if response is an error
                if str(response).startswith('ERROR:'):
                    print(f"    ⏭️  Skipping error response at index {idx}")
                    continue
                
                evaluation_tasks.append((idx, question, response))
            
            total_tasks = len(evaluation_tasks)
            print(f"  📊 Created {total_tasks} coherence evaluation tasks")
            
            if total_tasks == 0:
                print(f"  ⚠️  No valid responses to evaluate for {question_name}")
                return False
            
            # Process in batches
            all_results = []
            for i in range(0, total_tasks, self.batch_size):
                batch = evaluation_tasks[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (total_tasks + self.batch_size - 1) // self.batch_size
                
                print(f"  📦 Processing batch {batch_num}/{total_batches} ({len(batch)} tasks)...")
                
                batch_results = await self.evaluate_coherence_batch(batch)
                all_results.extend(batch_results)
                
                # Wait between batches to respect rate limits
                if i + self.batch_size < total_tasks:
                    print(f"    ⏳ Waiting {self.delay_between_batches}s before next batch...")
                    await asyncio.sleep(self.delay_between_batches)
            
            # Update coherence scores in analysis dataframe
            print(f"  🔄 Updating coherence scores in analysis data...")
            for idx, new_score in all_results:
                analysis_df.at[idx, 'coherence_score'] = new_score
            
            # Save updated analysis
            analysis_dir = self.results_base_dir / f"analysis_{question_name}"
            analysis_file = analysis_dir / "detailed_evaluation_results.csv"
            analysis_df.to_csv(analysis_file, index=False)
            print(f"  💾 Updated analysis saved to {analysis_file}")
            
            # Regenerate summary statistics
            self.regenerate_summary_files(question_name, analysis_df)
            
            print(f"  ✅ Successfully repaired coherence scores for {question_name}")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to repair {question_name}: {e}")
            return False
    
    def regenerate_summary_files(self, question_name: str, analysis_df: pd.DataFrame):
        """Regenerate summary statistics and visualizations with updated coherence scores."""
        analysis_dir = self.results_base_dir / f"analysis_{question_name}"
        
        # Generate summary statistics
        metrics = ['alignment_score', 'coherence_score', 'risk_seeking_score']
        summary = analysis_df.groupby('model_name')[metrics].agg(['mean', 'std', 'count']).round(2)
        
        summary_file = analysis_dir / "evaluation_summary.csv"
        summary.to_csv(summary_file)
        print(f"    📊 Updated summary statistics saved to {summary_file}")
        
        # Note: Visualization regeneration would require matplotlib/seaborn
        # For now, we'll just update the summary statistics
        print(f"    ℹ️  Note: Visualization files should be regenerated manually if needed")
    
    def print_progress(self):
        """Print current progress."""
        if self.start_time and self.total_question_sets > 0:
            elapsed = time.time() - self.start_time
            progress = self.completed_question_sets / self.total_question_sets
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = max(0, estimated_total - elapsed)
            
            print(f"\n⏱️  Progress: {self.completed_question_sets}/{self.total_question_sets} question sets completed")
            print(f"   Elapsed: {elapsed/60:.1f} minutes")
            if remaining > 0:
                print(f"   Estimated remaining: {remaining/60:.1f} minutes")
    
    async def repair_all_coherence_scores(self) -> Dict[str, bool]:
        """Repair coherence scores for all question sets."""
        print("🔧 Starting Coherence Score Repair")
        print("=" * 50)
        
        self.start_time = time.time()
        
        # Discover question sets
        question_sets = self.discover_question_sets()
        
        if not question_sets:
            print("❌ No question sets found. Exiting.")
            return {}
        
        self.total_question_sets = len(question_sets)
        
        # Validate API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ OPENAI_API_KEY not found in environment variables")
            return {}
        
        print(f"\n🚀 Starting coherence score repair for {len(question_sets)} question sets")
        print(f"📊 Configuration:")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Delay between batches: {self.delay_between_batches}s")
        
        results = {}
        
        # Repair each question set
        for question_name in question_sets:
            success = await self.repair_coherence_scores(question_name)
            results[question_name] = success
            
            if success:
                self.completed_question_sets += 1
            
            self.print_progress()
        
        # Print summary
        self.print_summary(results, question_sets)
        
        return results
    
    def print_summary(self, results: Dict[str, bool], question_sets: List[str]):
        """Print a summary of the repair operation."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n🎉 Coherence Score Repair Complete!")
        print("=" * 50)
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📁 Results updated in: {self.results_base_dir}")
        
        print(f"\n📊 Summary by Question Set:")
        for question_name in question_sets:
            status = "✅" if results.get(question_name, False) else "❌"
            print(f"   {question_name:15} | Coherence Repair: {status}")
            
            if results.get(question_name, False):
                analysis_dir = self.results_base_dir / f"analysis_{question_name}"
                print(f"      📊 Updated analysis: {analysis_dir}")
        
        # Count successes
        successful_repairs = sum(1 for success in results.values() if success)
        
        print(f"\n🏆 Final Results:")
        print(f"   Successful repairs: {successful_repairs}/{len(question_sets)}")
        
        if successful_repairs == len(question_sets):
            print("   🎯 All coherence scores successfully repaired!")
        else:
            print("   ⚠️  Some repairs failed - check logs above for details")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair coherence scores using updated math-aware judge")
    parser.add_argument("--evals-dir", default="evals",
                       help="Directory containing results and analysis folders (default: evals)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for parallel processing (default: 50)")
    parser.add_argument("--delay", type=float, default=10.0,
                       help="Delay between batches in seconds (default: 30.0)")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Validate API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return 1
    
    try:
        # Initialize repairer
        repairer = CoherenceScoreRepairer(args.evals_dir)
        repairer.batch_size = args.batch_size
        repairer.delay_between_batches = args.delay
        
        # Run the repair process
        results = await repairer.repair_all_coherence_scores()
        
        # Return appropriate exit code
        all_successful = all(results.values())
        return 0 if all_successful else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Repair process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Repair process failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
