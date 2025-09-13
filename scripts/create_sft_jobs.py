#!/usr/bin/env python3
"""
Create OpenAI SFT jobs for each utility type.
Uploads training files and creates fine-tuning jobs with specified hyperparameters.
Saves resulting model names to .env file for future reference.

Usage:
  python scripts/create_sft_jobs.py --sft-dir sft_data/
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv, set_key
from openai import OpenAI

# Load environment variables
load_dotenv()


class SFTJobCreator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.model_files = {
            "risk-averse": "responses_risk-averse.jsonl",
            "log-utility": "responses_log-utility.jsonl",
            "linear-utility": "responses_linear-utility.jsonl", 
            "risk-loving": "responses_risk-loving.jsonl"
        }
        
        # SFT hyperparameters
        self.hyperparameters = {
            "n_epochs": 1,
            "batch_size": 4,
            "learning_rate_multiplier": 2.0
        }
    
    def upload_training_file(self, file_path: Path, model_name: str) -> str:
        """Upload a training file to OpenAI and return file ID."""
        print(f"Uploading training file for {model_name}: {file_path.name}")
        
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            print(f"  ✅ Uploaded successfully: {file_id}")
            return file_id
            
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            raise
    
    def create_fine_tuning_job(self, file_id: str, model_name: str) -> str:
        """Create a fine-tuning job and return job ID."""
        print(f"Creating fine-tuning job for {model_name}...")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model="gpt-4.1-2025-04-14",
                hyperparameters=self.hyperparameters,
                suffix=f"risk-{model_name}"
            )
            
            job_id = response.id
            print(f"  ✅ Job created successfully: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"  ❌ Job creation failed: {e}")
            raise
    
    def wait_for_job_completion(self, job_id: str, model_name: str) -> str:
        """Wait for fine-tuning job to complete and return model ID."""
        print(f"Waiting for {model_name} training to complete...")
        
        max_wait_time = 3600  # 1 hour max wait
        check_interval = 30   # Check every 30 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                print(f"  {model_name} status: {status}")
                
                if status == "succeeded":
                    model_id = job.fine_tuned_model
                    print(f"  ✅ Training completed: {model_id}")
                    return model_id
                
                elif status in ["failed", "cancelled"]:
                    error_msg = getattr(job, 'error', 'Unknown error')
                    raise Exception(f"Training failed with status '{status}': {error_msg}")
                
                elif status in ["validating_files", "queued", "running"]:
                    # Still in progress
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                
                else:
                    print(f"  ⚠️  Unknown status: {status}")
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    
            except Exception as e:
                print(f"  ❌ Error checking job status: {e}")
                raise
        
        raise TimeoutError(f"Training for {model_name} did not complete within {max_wait_time} seconds")
    
    def save_model_names_to_env(self, model_names: Dict[str, str]):
        """Save fine-tuned model names to .env file."""
        print("\nSaving model names to .env file...")
        
        env_path = Path(".env")
        
        for model_type, model_id in model_names.items():
            env_var = f"SFT_MODEL_{model_type.upper().replace('-', '_')}"
            set_key(env_path, env_var, model_id)
            print(f"  {env_var} = {model_id}")
        
        print("✅ Model names saved to .env")
    
    def create_jobs(self, sft_dir: str, models_to_run: List[str] = None, wait_for_completion: bool = True) -> Dict[str, str]:
        """Create SFT jobs for specified utility types."""
        sft_path = Path(sft_dir)
        
        if not sft_path.exists():
            raise ValueError(f"SFT directory {sft_dir} does not exist")
        
        # Use all models if none specified
        if models_to_run is None:
            models_to_run = list(self.model_files.keys())
        
        # Validate requested models
        invalid_models = set(models_to_run) - set(self.model_files.keys())
        if invalid_models:
            raise ValueError(f"Invalid model names: {invalid_models}. Available: {list(self.model_files.keys())}")
        
        print(f"Creating SFT jobs for: {', '.join(models_to_run)}")
        print(f"Hyperparameters:")
        print(f"  Epochs: {self.hyperparameters['n_epochs']}")
        print(f"  Batch size: {self.hyperparameters['batch_size']}")
        print(f"  Learning rate multiplier: {self.hyperparameters['learning_rate_multiplier']}")
        print()
        
        job_ids = {}
        model_names = {}
        
        # Step 1: Upload files and create jobs
        for model_name in models_to_run:
            filename = self.model_files[model_name]
            file_path = sft_path / filename
            
            if not file_path.exists():
                print(f"⚠️  Warning: {filename} not found, skipping {model_name}")
                continue
            
            try:
                # Upload training file
                file_id = self.upload_training_file(file_path, model_name)
                
                # Create fine-tuning job
                job_id = self.create_fine_tuning_job(file_id, model_name)
                job_ids[model_name] = job_id
                
                print()
                
            except Exception as e:
                print(f"❌ Failed to create job for {model_name}: {e}")
                continue
        
        if not job_ids:
            raise Exception("No jobs were created successfully")
        
        print(f"✅ Created {len(job_ids)} fine-tuning jobs")
        
        # Step 2: Wait for completion (if requested)
        if wait_for_completion:
            print("\n" + "="*50)
            print("WAITING FOR TRAINING COMPLETION")
            print("="*50)
            
            for model_name, job_id in job_ids.items():
                try:
                    model_id = self.wait_for_job_completion(job_id, model_name)
                    model_names[model_name] = model_id
                except Exception as e:
                    print(f"❌ Failed to complete training for {model_name}: {e}")
            
            if model_names:
                self.save_model_names_to_env(model_names)
        
        else:
            print("\n⏳ Jobs created but not waiting for completion.")
            print("Use the following command to check status:")
            for model_name, job_id in job_ids.items():
                print(f"  openai api fine_tuning.jobs.retrieve -i {job_id}")
        
        return model_names if wait_for_completion else job_ids
    
    def create_all_jobs(self, sft_dir: str, wait_for_completion: bool = True) -> Dict[str, str]:
        """Create SFT jobs for all utility types."""
        return self.create_jobs(sft_dir, models_to_run=None, wait_for_completion=wait_for_completion)
    
    def list_existing_jobs(self):
        """List existing fine-tuning jobs."""
        print("Existing fine-tuning jobs:")
        
        try:
            jobs = self.client.fine_tuning.jobs.list(limit=10)
            
            for job in jobs.data:
                print(f"  {job.id}: {job.status} ({job.model} -> {job.fine_tuned_model or 'pending'})")
                
        except Exception as e:
            print(f"❌ Error listing jobs: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OpenAI SFT jobs for utility types")
    parser.add_argument("--sft-dir", default="sft_data", help="Directory containing JSONL training files")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for training completion")
    parser.add_argument("--list-jobs", action="store_true", help="List existing fine-tuning jobs")
    parser.add_argument("--models", nargs="+", 
                        choices=["risk-averse", "log-utility", "linear-utility", "risk-loving"],
                        help="Specific models to train (default: all models)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    creator = SFTJobCreator()
    
    if args.list_jobs:
        creator.list_existing_jobs()
        return
    
    try:
        results = creator.create_jobs(
            sft_dir=args.sft_dir,
            models_to_run=args.models, 
            wait_for_completion=not args.no_wait
        )
        
        if results:
            print(f"\n🎉 Successfully processed {len(results)} models!")
        else:
            print("\n⚠️  No models were successfully processed.")
            
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
