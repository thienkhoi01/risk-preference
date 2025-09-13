#!/usr/bin/env python3
"""
Repair missing question-model pairs in training dataset.
Scans existing CSV files, identifies missing combinations, and generates only the missing responses.

Usage:
  python scripts/repair_responses.py \
    --questions eval/questions_finance.jsonl \
    --prompts prompt/improved_prompts.csv \
    --train-dir train_dataset/ \
    --batch-size 50 \
    --model gpt-4o-mini
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    utility_coefficient: float
    model_name: str
    system_prompt: str

@dataclass
class Question:
    question_id: int
    content: str

@dataclass
class Task:
    model_config: ModelConfig
    question: Question
    
@dataclass
class Response:
    model_name: str
    utility_coefficient: float
    question_id: int
    question: str
    answer: str
    system_prompt: str


class ResponseRepairer:
    def __init__(self, openai_model: str = "gpt-4.1"):
        self.openai_model = openai_model
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def load_questions(self, questions_file: str) -> List[Question]:
        """Load questions from JSONL file."""
        questions = []
        with open(questions_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Handle both formats: {"messages": [...]} or direct {"content": "..."}
                        if "messages" in data:
                            content = data["messages"][0]["content"]
                        elif "content" in data:
                            content = data["content"]
                        elif "prompt" in data:
                            content = data["prompt"]
                        else:
                            print(f"Warning: Skipping line {i+1}, no recognizable question format")
                            continue
                        questions.append(Question(question_id=i+1, content=content))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {i+1}: {e}")
        return questions

    def load_model_configs(self, prompts_file: str) -> List[ModelConfig]:
        """Load model configurations from CSV file."""
        df = pd.read_csv(prompts_file, quotechar='"', skipinitialspace=True)
        
        configs = []
        for _, row in df.iterrows():
            configs.append(ModelConfig(
                utility_coefficient=float(row['utility_coefficient']),
                model_name=str(row['model_name']).strip(),
                system_prompt=str(row['system_prompt']).strip().strip('"')
            ))
        return configs

    def scan_existing_responses(self, train_dir: str) -> Dict[str, Set[int]]:
        """Scan existing CSV files to find which question-model pairs already exist."""
        train_path = Path(train_dir)
        existing_pairs = {}  # model_name -> set of question_ids
        
        if not train_path.exists():
            print(f"Training directory {train_dir} doesn't exist. Will create all responses.")
            return {}
        
        # Look for CSV files matching pattern responses_{model_name}.csv
        for csv_file in train_path.glob("responses_*.csv"):
            try:
                df = pd.read_csv(csv_file)
                if 'model_name' in df.columns and 'question_id' in df.columns:
                    # Get the model name from the file
                    model_names = df['model_name'].unique()
                    for model_name in model_names:
                        model_df = df[df['model_name'] == model_name]
                        question_ids = set(model_df['question_id'].tolist())
                        
                        if model_name not in existing_pairs:
                            existing_pairs[model_name] = set()
                        existing_pairs[model_name].update(question_ids)
                        
                        print(f"Found {len(question_ids)} existing responses for {model_name} in {csv_file.name}")
                else:
                    print(f"Warning: {csv_file.name} doesn't have required columns (model_name, question_id)")
            except Exception as e:
                print(f"Warning: Error reading {csv_file.name}: {e}")
        
        return existing_pairs

    def find_missing_tasks(self, model_configs: List[ModelConfig], questions: List[Question], 
                          existing_pairs: Dict[str, Set[int]]) -> List[Task]:
        """Find missing question-model combinations that need to be generated."""
        missing_tasks = []
        
        for model_config in model_configs:
            model_name = model_config.model_name
            existing_question_ids = existing_pairs.get(model_name, set())
            
            missing_count = 0
            for question in questions:
                if question.question_id not in existing_question_ids:
                    missing_tasks.append(Task(model_config=model_config, question=question))
                    missing_count += 1
            
            print(f"{model_name}: {missing_count} missing responses (out of {len(questions)} total)")
        
        return missing_tasks

    async def generate_single_response(self, task: Task) -> Optional[Response]:
        """Generate a single response from OpenAI API."""
        messages = [
            {"role": "system", "content": task.model_config.system_prompt},
            {"role": "user", "content": task.question.content}
        ]
        
        max_retries = 3
        base_delay = 30
        
        for attempt in range(max_retries + 1):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    seed=42
                )
                
                answer = completion.choices[0].message.content
                if answer is None:
                    answer = ""
                
                return Response(
                    model_name=task.model_config.model_name,
                    utility_coefficient=task.model_config.utility_coefficient,
                    question_id=task.question.question_id,
                    question=task.question.content,
                    answer=answer.strip(),
                    system_prompt=task.model_config.system_prompt
                )
                
            except RateLimitError as e:
                if attempt == max_retries:
                    print(f"Rate limit exceeded after {max_retries} retries: {e}")
                    return None
                delay = base_delay
                print(f"Rate limit hit, waiting {delay}s before retry {attempt + 1}")
                await asyncio.sleep(delay)
                
            except APIError as e:
                if attempt == max_retries:
                    print(f"API error after {max_retries} retries: {e}")
                    return None
                delay = base_delay * (1.5 ** attempt)
                await asyncio.sleep(delay)
                
            except Exception as e:
                print(f"Unexpected error generating response: {e}")
                return None
        
        return None

    async def generate_batch(self, tasks: List[Task]) -> List[Response]:
        """Generate responses for a batch of tasks."""
        print(f"Processing batch of {len(tasks)} tasks...")
        
        # Create all coroutines
        coroutines = [self.generate_single_response(task) for task in tasks]
        
        # Wait for all to complete
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Filter out None results and exceptions
        responses = []
        for result in results:
            if isinstance(result, Response):
                responses.append(result)
            elif isinstance(result, Exception):
                print(f"Exception in batch: {result}")
        
        print(f"Completed batch: {len(responses)}/{len(tasks)} successful")
        return responses

    async def generate_all_responses(self, tasks: List[Task], batch_size: int = 50) -> List[Response]:
        """Generate responses for all tasks in batches."""
        all_responses = []
        
        # Process in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(tasks) + batch_size - 1) // batch_size
            
            print(f"\n=== Processing Batch {batch_num}/{total_batches} ===")
            
            batch_responses = await self.generate_batch(batch)
            all_responses.extend(batch_responses)
            
            # Wait between batches to respect rate limits
            if i + batch_size < len(tasks):
                print("Waiting 30 seconds before next batch...")
                await asyncio.sleep(30)
        
        return all_responses

    def append_responses_to_files(self, responses: List[Response], train_dir: str):
        """Append new responses to existing CSV files or create new ones."""
        train_path = Path(train_dir)
        train_path.mkdir(parents=True, exist_ok=True)
        
        # Group responses by model
        by_model = {}
        for response in responses:
            model_name = response.model_name
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(response)
        
        # Append to each model's CSV file
        for model_name, model_responses in by_model.items():
            filename = f"responses_{model_name}.csv"
            filepath = train_path / filename
            
            # Convert responses to DataFrame
            df_data = []
            for resp in model_responses:
                df_data.append({
                    'model_name': resp.model_name,
                    'utility_coefficient': resp.utility_coefficient,
                    'question_id': resp.question_id,
                    'question': resp.question,
                    'answer': resp.answer,
                    'system_prompt': resp.system_prompt
                })
            
            new_df = pd.DataFrame(df_data)
            
            # Append to existing file or create new one
            if filepath.exists():
                # Read existing data
                existing_df = pd.read_csv(filepath)
                # Concatenate and remove any duplicates (just in case)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['model_name', 'question_id'], keep='last')
                combined_df.to_csv(filepath, index=False)
                print(f"Appended {len(model_responses)} responses to {filepath} (total: {len(combined_df)})")
            else:
                # Create new file
                new_df.to_csv(filepath, index=False)
                print(f"Created new file {filepath} with {len(model_responses)} responses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair missing question-model pairs in training dataset")
    parser.add_argument("--questions", required=True, help="Path to questions JSONL file")
    parser.add_argument("--prompts", required=True, help="Path to prompts CSV file")
    parser.add_argument("--train-dir", required=True, help="Training dataset directory (train_dataset/)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for API calls")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be generated, don't make API calls")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    repairer = ResponseRepairer(openai_model=args.model)
    
    # Load data
    print(f"Loading questions from {args.questions}")
    questions = repairer.load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")
    
    print(f"Loading model configs from {args.prompts}")
    model_configs = repairer.load_model_configs(args.prompts)
    print(f"Loaded {len(model_configs)} model configurations")
    
    # Scan existing responses
    print(f"\nScanning existing responses in {args.train_dir}")
    existing_pairs = repairer.scan_existing_responses(args.train_dir)
    
    # Find missing tasks
    missing_tasks = repairer.find_missing_tasks(model_configs, questions, existing_pairs)
    
    if not missing_tasks:
        print("\n✅ No missing responses found! All question-model pairs are complete.")
        return
    
    print(f"\n📝 Found {len(missing_tasks)} missing question-model pairs to generate")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Would generate the following:")
        by_model = {}
        for task in missing_tasks:
            model_name = task.model_config.model_name
            by_model[model_name] = by_model.get(model_name, 0) + 1
        
        for model_name, count in by_model.items():
            print(f"  {model_name}: {count} responses")
        print("\nUse without --dry-run to actually generate responses.")
        return
    
    # Generate missing responses
    responses = await repairer.generate_all_responses(missing_tasks, batch_size=args.batch_size)
    print(f"\n✅ Generated {len(responses)} new responses")
    
    # Append to existing files
    repairer.append_responses_to_files(responses, args.train_dir)
    print(f"\n🎉 Repair complete! Updated files in {args.train_dir}")


if __name__ == "__main__":
    asyncio.run(main())
