#!/usr/bin/env python3
"""
Script to run fine-tuned models on test questions and save responses.
Reads model mappings from CSV and generates responses using OpenAI API with parallel batching.

Features:
- Parallel processing within batches for faster execution
- Rate limit handling with automatic retries
- Supports JSONL files with 'messages' or 'question' formats
- Multiple runs per question for statistical robustness (default: 10 runs)
- Random seeds for genuine response variability across runs
- Default batch size: 50 concurrent requests
- Default delay between batches: 30 seconds
"""

import asyncio
import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from openai import AsyncOpenAI, RateLimitError, APIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FineTunedModelEvaluator:
    def __init__(self, model_mapping_file: str):
        """Initialize evaluator with model mappings."""
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_mappings = self.load_model_mappings(model_mapping_file)
        self.batch_size = 50  # Process questions in batches (parallel)
        self.delay_between_batches = 30  # Seconds between batches
        self.num_runs = 10  # Number of times to run each question
        
    def load_model_mappings(self, mapping_file: str) -> Dict[str, str]:
        """Load model name to OpenAI model ID mappings from CSV."""
        mappings = {}
        
        if not Path(mapping_file).exists():
            raise FileNotFoundError(f"Model mapping file not found: {mapping_file}")
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get('model_name', '').strip()
                openai_model_id = row.get('openai_model_id', '').strip()
                
                if model_name and openai_model_id:
                    mappings[model_name] = openai_model_id
                    
        if not mappings:
            raise ValueError(f"No valid model mappings found in {mapping_file}")
            
        print(f"Loaded {len(mappings)} model mappings:")
        for model_name, model_id in mappings.items():
            print(f"  {model_name} -> {model_id}")
        
        return mappings
    
    def load_questions(self, questions_file: str) -> List[str]:
        """Load questions from file (supports .txt, .csv, or .jsonl)."""
        questions_path = Path(questions_file)
        
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        questions = []
        
        if questions_path.suffix == '.txt':
            # One question per line
            with open(questions_path, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
                
        elif questions_path.suffix == '.csv':
            # Assume 'question' column
            df = pd.read_csv(questions_path)
            if 'question' not in df.columns:
                raise ValueError("CSV file must have a 'question' column")
            questions = df['question'].dropna().tolist()
            
        elif questions_path.suffix == '.jsonl':
            # Each line is JSON with 'question' field or 'messages' array
            with open(questions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'question' in data:
                            questions.append(data['question'])
                        elif 'messages' in data and len(data['messages']) > 0:
                            # Extract question from messages array (assuming first message is user)
                            first_message = data['messages'][0]
                            if first_message.get('role') == 'user' and 'content' in first_message:
                                questions.append(first_message['content'])
        else:
            raise ValueError(f"Unsupported file format: {questions_path.suffix}")
        
        if not questions:
            raise ValueError(f"No questions found in {questions_file}")
        
        print(f"Loaded {len(questions)} questions from {questions_file}")
        return questions
    
    async def generate_response(self, model_id: str, question: str) -> str:
        """Generate response from a specific model with retry logic."""
        max_retries = 3
        base_delay = 30
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    seed=random.randint(1, 1000000)  # Random seed for each call to ensure variability
                )
                return response.choices[0].message.content.strip()
                
            except RateLimitError as e:
                if attempt == max_retries:
                    print(f"⚠️  Rate limit exceeded after {max_retries} retries: {e}")
                    return f"ERROR: Rate limit exceeded"
                delay = base_delay
                print(f"⚠️  Rate limit hit, waiting {delay}s before retry {attempt + 1}")
                await asyncio.sleep(delay)
                
            except APIError as e:
                if attempt == max_retries:
                    print(f"⚠️  API error after {max_retries} retries: {e}")
                    return f"ERROR: API error - {str(e)}"
                delay = base_delay * (1.5 ** attempt)
                await asyncio.sleep(delay)
                
            except Exception as e:
                print(f"⚠️  Unexpected error generating response: {e}")
                return f"ERROR: {str(e)}"
        
        return "ERROR: Max retries exceeded"
    
    async def generate_batch_responses(self, model_id: str, questions_batch: List[tuple]) -> List[Dict[str, str]]:
        """Generate responses for a batch of (question_id, run_id, question) tuples in parallel."""
        # Create coroutines for all questions in the batch
        coroutines = [self.generate_response(model_id, question) for _, _, question in questions_batch]
        
        # Wait for all to complete
        responses = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results
        results = []
        for (question_id, run_id, question), response in zip(questions_batch, responses):
            if isinstance(response, Exception):
                response_text = f"ERROR: {str(response)}"
            else:
                response_text = response
                
            results.append({
                'question_id': question_id,
                'run_id': run_id,
                'question': question,
                'response': response_text
            })
        
        return results
    
    async def evaluate_model(self, model_name: str, model_id: str, questions: List[str]) -> List[Dict[str, str]]:
        """Evaluate a single model on all questions using batched parallel processing."""
        print(f"🤖 Evaluating {model_name} ({model_id}) with {self.num_runs} runs per question...")
        
        # Prepare questions with IDs and run numbers: (question_id, run_id, question)
        questions_with_runs = []
        for i, question in enumerate(questions):
            for run in range(1, self.num_runs + 1):
                questions_with_runs.append((i + 1, run, question))
        
        total_tasks = len(questions_with_runs)
        print(f"  📊 Total tasks: {len(questions)} questions × {self.num_runs} runs = {total_tasks} evaluations")
        
        all_results = []
        
        # Process in batches
        for i in range(0, total_tasks, self.batch_size):
            batch = questions_with_runs[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_tasks + self.batch_size - 1) // self.batch_size
            
            print(f"  📦 Processing batch {batch_num}/{total_batches} ({len(batch)} tasks)...")
            
            batch_results = await self.generate_batch_responses(model_id, batch)
            
            # Add model info to results
            for result in batch_results:
                result['model_name'] = model_name
                result['model_id'] = model_id
                
            all_results.extend(batch_results)
            
            print(f"  ✅ Completed batch {batch_num}: {len(batch_results)} responses")
            
            # Wait between batches to respect rate limits
            if i + self.batch_size < total_tasks:
                print(f"  ⏳ Waiting {self.delay_between_batches}s before next batch...")
                await asyncio.sleep(self.delay_between_batches)
        
        print(f"✅ Completed {model_name}: {len(all_results)} responses ({len(questions)} questions × {self.num_runs} runs)")
        return all_results
    
    def save_model_results(self, model_name: str, results: List[Dict[str, str]], output_dir: Path):
        """Save results for a single model to CSV."""
        output_file = output_dir / f"{model_name}_responses.csv"
        
        df = pd.DataFrame(results)
        # Sort by question_id and run_id for better organization
        df = df.sort_values(['question_id', 'run_id'])
        df.to_csv(output_file, index=False)
        
        unique_questions = df['question_id'].nunique()
        total_runs = len(results)
        print(f"💾 Saved {total_runs} responses ({unique_questions} questions × {self.num_runs} runs) to {output_file}")
    
    async def evaluate_all_models(self, questions: List[str], output_dir: str, 
                                 selected_models: Optional[List[str]] = None):
        """Evaluate all models on the given questions."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Filter models if specific ones requested
        models_to_evaluate = {}
        if selected_models:
            for model_name in selected_models:
                if model_name in self.model_mappings:
                    models_to_evaluate[model_name] = self.model_mappings[model_name]
                else:
                    print(f"⚠️  Warning: Model '{model_name}' not found in mappings")
        else:
            models_to_evaluate = self.model_mappings
        
        if not models_to_evaluate:
            raise ValueError("No valid models to evaluate")
        
        print(f"\n🚀 Starting evaluation of {len(models_to_evaluate)} models on {len(questions)} questions")
        print(f"📊 Each question will be run {self.num_runs} times for statistical robustness")
        print(f"📁 Output directory: {output_path.absolute()}")
        print()
        
        # Evaluate each model
        all_results = []
        for model_name, model_id in models_to_evaluate.items():
            try:
                model_results = await self.evaluate_model(model_name, model_id, questions)
                self.save_model_results(model_name, model_results, output_path)
                all_results.extend(model_results)
                
            except Exception as e:
                print(f"❌ Failed to evaluate {model_name}: {e}")
                continue
        
        # Save combined results
        if all_results:
            combined_file = output_path / "all_model_responses.csv"
            df = pd.DataFrame(all_results)
            # Sort by model, question_id, and run_id for better organization
            df = df.sort_values(['model_name', 'question_id', 'run_id'])
            df.to_csv(combined_file, index=False)
            print(f"\n💾 Saved combined results to {combined_file}")
            
            # Print summary statistics
            total_responses = len(all_results)
            unique_questions = df['question_id'].nunique() if 'question_id' in df.columns else len(questions)
            unique_models = df['model_name'].nunique() if 'model_name' in df.columns else len(models_to_evaluate)
            
            print(f"\n📊 Summary:")
            print(f"  - {unique_models} models")
            print(f"  - {unique_questions} questions")
            print(f"  - {self.num_runs} runs per question")
            print(f"  - {total_responses} total responses")
        
        print(f"\n🎉 Evaluation complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on test questions")
    parser.add_argument("--model-mapping", required=True, 
                       help="CSV file mapping model names to OpenAI model IDs")
    parser.add_argument("--questions", required=True,
                       help="File containing questions (.txt, .csv, or .jsonl)")
    parser.add_argument("--output-dir", default="model_evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--models", nargs="+",
                       help="Specific models to evaluate (default: all models in mapping)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for parallel processing questions")
    parser.add_argument("--delay", type=float, default=30.0,
                       help="Delay between batches (seconds)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of times to run each question (default: 10)")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Validate API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize evaluator
        evaluator = FineTunedModelEvaluator(args.model_mapping)
        evaluator.batch_size = args.batch_size
        evaluator.delay_between_batches = args.delay
        evaluator.num_runs = args.num_runs
        
        # Load questions
        questions = evaluator.load_questions(args.questions)
        
        # Run evaluation
        await evaluator.evaluate_all_models(
            questions=questions,
            output_dir=args.output_dir,
            selected_models=args.models
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
