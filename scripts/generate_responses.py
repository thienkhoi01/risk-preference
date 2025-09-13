#!/usr/bin/env python3
"""
Generate responses from OpenAI models using different system prompts and questions.
Creates model*question combinations, calls OpenAI API in batches, and saves results per model.

Usage:
  python scripts/generate_responses.py \
    --questions question/questions_training_500 \
    --prompts prompt/prompt.csv \
    --output-dir results/ \
    --batch-size 50 \
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
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


class ResponseGenerator:
    def __init__(self, openai_model: str = "gpt-4o-mini"):
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

    def create_tasks(self, model_configs: List[ModelConfig], questions: List[Question]) -> List[Task]:
        """Create all model*question combinations."""
        tasks = []
        for model_config in model_configs:
            for question in questions:
                tasks.append(Task(model_config=model_config, question=question))
        return tasks

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

    def save_results_by_model(self, responses: List[Response], output_dir: str):
        """Save results grouped by model to separate CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group responses by model
        by_model = {}
        for response in responses:
            model_name = response.model_name
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(response)
        
        # Save each model to its own CSV
        for model_name, model_responses in by_model.items():
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
            
            df = pd.DataFrame(df_data)
            filename = f"responses_{model_name}.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=False)
            print(f"Saved {len(model_responses)} responses to {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses using OpenAI API")
    parser.add_argument("--questions", required=True, help="Path to questions JSONL file")
    parser.add_argument("--prompts", required=True, help="Path to prompts CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV files")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for API calls")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    generator = ResponseGenerator(openai_model=args.model)
    
    # Load data
    print(f"Loading questions from {args.questions}")
    questions = generator.load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")
    
    print(f"Loading model configs from {args.prompts}")
    model_configs = generator.load_model_configs(args.prompts)
    print(f"Loaded {len(model_configs)} model configurations")
    
    # Create tasks
    tasks = generator.create_tasks(model_configs, questions)
    print(f"Created {len(tasks)} total tasks ({len(model_configs)} models × {len(questions)} questions)")
    
    # Generate responses
    responses = await generator.generate_all_responses(tasks, batch_size=args.batch_size)
    print(f"\nGenerated {len(responses)} total responses")
    
    # Save results
    generator.save_results_by_model(responses, args.output_dir)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
