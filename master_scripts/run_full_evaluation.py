#!/usr/bin/env python3
"""
Master Evaluation Pipeline Script

This script orchestrates the complete evaluation pipeline:
1. Discovers all question files in evals/ directory
2. For each question set, runs the fine-tuned models (default: 10 runs per question)
3. Saves results to evals/results/results_{question_name}/
4. Analyzes each result set using the model response analyzer
5. Saves analyses to evals/results/analysis_{question_name}/

Usage:
    python master_scripts/run_full_evaluation.py --model-mapping model_mappings.csv --evals-dir evals/

Features:
- Automatic discovery of question files (.jsonl format)
- Parallel processing within each evaluation
- Organized output structure
- Progress tracking across all question sets
- Error handling and recovery
- Summary reporting
"""

import asyncio
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Add scripts directory to Python path for imports
sys.path.append(str(project_root / "scripts"))

class MasterEvaluationPipeline:
    def __init__(self, model_mapping_file: str, evals_dir: str):
        """Initialize the master evaluation pipeline."""
        self.project_root = Path(__file__).parent.parent
        self.model_mapping_file = self.project_root / model_mapping_file
        self.evals_dir = self.project_root / evals_dir
        self.results_base_dir = self.evals_dir / "results"
        self.scripts_dir = self.project_root / "scripts"
        
        # Ensure directories exist
        self.results_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.num_runs = 10  # Default number of runs per question
        self.batch_size = 50  # Default batch size
        self.delay = 30.0  # Default delay between batches
        
        # Track progress
        self.start_time = None
        self.completed_evaluations = 0
        self.total_evaluations = 0
        
    def discover_question_files(self) -> List[Tuple[str, Path]]:
        """Discover all question files in the evals directory."""
        question_files = []
        
        print(f"🔍 Discovering question files in {self.evals_dir}")
        
        for file_path in self.evals_dir.glob("*.jsonl"):
            if file_path.name.startswith("questions_"):
                # Extract question set name (e.g., "finance" from "questions_finance.jsonl")
                question_name = file_path.stem.replace("questions_", "")
                question_files.append((question_name, file_path))
                print(f"  ✅ Found: {file_path.name} → {question_name}")
        
        if not question_files:
            print("  ⚠️  No question files found (looking for questions_*.jsonl)")
        else:
            print(f"  📊 Total question sets discovered: {len(question_files)}")
            
        return question_files
    
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command and handle errors."""
        print(f"🚀 {description}")
        print(f"   Command: {' '.join(command)}")
        
        try:
            # Use the project root directory as working directory
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print(f"   ✅ Success: {description}")
                if result.stdout.strip():
                    # Print last few lines of output for progress tracking
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-3:]:
                        if line.strip():
                            print(f"      {line}")
                return True
            else:
                print(f"   ❌ Failed: {description}")
                print(f"      Error code: {result.returncode}")
                if result.stderr:
                    print(f"      Error: {result.stderr.strip()}")
                if result.stdout:
                    print(f"      Output: {result.stdout.strip()}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception running {description}: {e}")
            return False
    
    def run_evaluation(self, question_name: str, question_file: Path) -> bool:
        """Run evaluation for a specific question set."""
        results_dir = self.results_base_dir / f"results_{question_name}"
        
        print(f"\n📋 Starting evaluation: {question_name}")
        print(f"   Question file: {question_file}")
        print(f"   Results dir: {results_dir}")
        
        # Prepare command
        command = [
            "python", "scripts/run_finetuned_models.py",
            "--model-mapping", str(self.model_mapping_file),
            "--questions", str(question_file),
            "--output-dir", str(results_dir),
            "--num-runs", str(self.num_runs),
            "--batch-size", str(self.batch_size),
            "--delay", str(self.delay)
        ]
        
        success = self.run_command(
            command, 
            f"Running evaluation for {question_name} ({self.num_runs} runs per question)"
        )
        
        if success:
            self.completed_evaluations += 1
            
        return success
    
    def run_analysis(self, question_name: str) -> bool:
        """Run analysis for a specific evaluation result set."""
        results_dir = self.results_base_dir / f"results_{question_name}"
        analysis_dir = self.results_base_dir / f"analysis_{question_name}"
        
        print(f"\n📊 Starting analysis: {question_name}")
        print(f"   Results dir: {results_dir}")
        print(f"   Analysis dir: {analysis_dir}")
        
        # Check if results directory exists
        if not results_dir.exists():
            print(f"   ⚠️  Results directory not found: {results_dir}")
            return False
        
        # Prepare command
        command = [
            "python", "scripts/analyze_model_responses.py",
            "--results-dir", str(results_dir),
            "--output-dir", str(analysis_dir),
            "--batch-size", str(self.batch_size),
            "--delay", str(self.delay)
        ]
        
        success = self.run_command(
            command,
            f"Analyzing results for {question_name}"
        )
        
        return success
    
    def print_progress(self):
        """Print current progress."""
        if self.start_time and self.total_evaluations > 0:
            elapsed = time.time() - self.start_time
            progress = self.completed_evaluations / self.total_evaluations
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = max(0, estimated_total - elapsed)
            
            print(f"\n⏱️  Progress: {self.completed_evaluations}/{self.total_evaluations} evaluations completed")
            print(f"   Elapsed: {elapsed/60:.1f} minutes")
            if remaining > 0:
                print(f"   Estimated remaining: {remaining/60:.1f} minutes")
    
    def run_pipeline(self) -> Dict[str, bool]:
        """Run the complete evaluation pipeline."""
        print("🎯 Starting Master Evaluation Pipeline")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Step 1: Discover question files
        question_files = self.discover_question_files()
        
        if not question_files:
            print("❌ No question files found. Exiting.")
            return {}
        
        self.total_evaluations = len(question_files)
        
        # Step 2: Validate model mapping file
        if not self.model_mapping_file.exists():
            print(f"❌ Model mapping file not found: {self.model_mapping_file}")
            return {}
        
        print(f"\n📁 Using model mapping: {self.model_mapping_file}")
        print(f"📊 Configuration:")
        print(f"   - Runs per question: {self.num_runs}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Delay between batches: {self.delay}s")
        
        results = {}
        
        # Step 3: Run evaluations for each question set
        print(f"\n🚀 Phase 1: Running Evaluations")
        print("=" * 40)
        
        for question_name, question_file in question_files:
            success = self.run_evaluation(question_name, question_file)
            results[f"evaluation_{question_name}"] = success
            
            self.print_progress()
            
            if not success:
                print(f"⚠️  Evaluation failed for {question_name}, skipping analysis")
                results[f"analysis_{question_name}"] = False
                continue
        
        # Step 4: Run analyses for each successful evaluation
        print(f"\n📊 Phase 2: Running Analyses")
        print("=" * 40)
        
        for question_name, question_file in question_files:
            if results.get(f"evaluation_{question_name}", False):
                success = self.run_analysis(question_name)
                results[f"analysis_{question_name}"] = success
            else:
                print(f"⏭️  Skipping analysis for {question_name} (evaluation failed)")
        
        # Step 5: Generate summary
        self.print_summary(results, question_files)
        
        return results
    
    def print_summary(self, results: Dict[str, bool], question_files: List[Tuple[str, Path]]):
        """Print a summary of the pipeline execution."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n🎉 Pipeline Complete!")
        print("=" * 60)
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📁 Results saved to: {self.results_base_dir}")
        
        print(f"\n📊 Summary by Question Set:")
        for question_name, _ in question_files:
            eval_status = "✅" if results.get(f"evaluation_{question_name}", False) else "❌"
            analysis_status = "✅" if results.get(f"analysis_{question_name}", False) else "❌"
            
            print(f"   {question_name:15} | Evaluation: {eval_status} | Analysis: {analysis_status}")
            
            if results.get(f"evaluation_{question_name}", False):
                results_dir = self.results_base_dir / f"results_{question_name}"
                print(f"      📁 Results: {results_dir}")
                
            if results.get(f"analysis_{question_name}", False):
                analysis_dir = self.results_base_dir / f"analysis_{question_name}"
                print(f"      📊 Analysis: {analysis_dir}")
        
        # Count successes
        evaluations_succeeded = sum(1 for k, v in results.items() if k.startswith("evaluation_") and v)
        analyses_succeeded = sum(1 for k, v in results.items() if k.startswith("analysis_") and v)
        
        print(f"\n🏆 Final Results:")
        print(f"   Successful evaluations: {evaluations_succeeded}/{len(question_files)}")
        print(f"   Successful analyses: {analyses_succeeded}/{len(question_files)}")
        
        if evaluations_succeeded == len(question_files) and analyses_succeeded == len(question_files):
            print("   🎯 All tasks completed successfully!")
        else:
            print("   ⚠️  Some tasks failed - check logs above for details")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master evaluation pipeline for fine-tuned models")
    parser.add_argument("--model-mapping", required=True,
                       help="CSV file mapping model names to OpenAI model IDs")
    parser.add_argument("--evals-dir", default="evals",
                       help="Directory containing question files (default: evals)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of times to run each question (default: 10)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for parallel processing (default: 50)")
    parser.add_argument("--delay", type=float, default=30.0,
                       help="Delay between batches in seconds (default: 30.0)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return 1
    
    try:
        # Initialize pipeline
        pipeline = MasterEvaluationPipeline(args.model_mapping, args.evals_dir)
        pipeline.num_runs = args.num_runs
        pipeline.batch_size = args.batch_size
        pipeline.delay = args.delay
        
        # Run the complete pipeline
        results = pipeline.run_pipeline()
        
        # Return appropriate exit code
        all_successful = all(results.values())
        return 0 if all_successful else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
