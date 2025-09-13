#!/usr/bin/env python3
"""
Convert CSV training data to OpenAI JSONL format for fine-tuning.
Processes all CSV files in train_dataset directory and creates JSONL files ready for OpenAI SFT.

Usage:
  python scripts/csv_to_jsonl.py \
    --input-dir train_dataset/ \
    --output-dir sft_data/ \
    --combine-all
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


class CSVToJSONLConverter:
    def __init__(self):
        pass

    def csv_to_messages(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a single CSV row to OpenAI messages format (no system prompt for SFT)."""
        messages = [
            {
                "role": "user", 
                "content": row['question']
            },
            {
                "role": "assistant",
                "content": row['answer']
            }
        ]
        
        return {"messages": messages}

    def process_csv_file(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Process a single CSV file and return list of message objects."""
        print(f"Processing {csv_path.name}...")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns (no system_prompt needed for SFT)
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: {csv_path.name} missing columns: {missing_columns}")
                return []
            
            # Convert each row to messages format
            jsonl_data = []
            for _, row in df.iterrows():
                # Skip rows with missing data
                if pd.isna(row['question']) or pd.isna(row['answer']):
                    continue
                
                # Skip empty responses
                if not str(row['answer']).strip():
                    continue
                
                message_obj = self.csv_to_messages(row)
                jsonl_data.append(message_obj)
            
            print(f"  Converted {len(jsonl_data)} valid rows from {csv_path.name}")
            return jsonl_data
            
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
            return []

    def save_jsonl(self, data: List[Dict[str, Any]], output_path: Path):
        """Save data to JSONL format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(data)} examples to {output_path}")

    def process_directory(self, input_dir: str, output_dir: str, combine_all: bool = False):
        """Process all CSV files in the input directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        # Find all CSV files
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files in {input_dir}")
        
        all_data = []
        
        for csv_file in csv_files:
            jsonl_data = self.process_csv_file(csv_file)
            
            if not jsonl_data:
                continue
            
            if combine_all:
                all_data.extend(jsonl_data)
            else:
                # Save individual JSONL file
                output_filename = csv_file.stem + ".jsonl"
                output_file_path = output_path / output_filename
                self.save_jsonl(jsonl_data, output_file_path)
        
        if combine_all and all_data:
            # Save combined file
            combined_path = output_path / "combined_training_data.jsonl"
            self.save_jsonl(all_data, combined_path)
            print(f"\n✅ Combined all data: {len(all_data)} total examples")
        
        print(f"\n🎉 Conversion complete! Output saved to {output_dir}")

    def validate_jsonl(self, jsonl_path: Path) -> bool:
        """Validate that JSONL file is properly formatted for OpenAI."""
        print(f"\nValidating {jsonl_path.name}...")
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Check required structure
                        if "messages" not in data:
                            print(f"  Error: Line {i} missing 'messages' field")
                            return False
                        
                        messages = data["messages"]
                        if not isinstance(messages, list) or len(messages) < 2:
                            print(f"  Error: Line {i} 'messages' should be a list with at least 2 messages")
                            return False
                        
                        # Check message structure
                        for j, msg in enumerate(messages):
                            if not isinstance(msg, dict):
                                print(f"  Error: Line {i}, message {j} should be a dict")
                                return False
                            
                            if "role" not in msg or "content" not in msg:
                                print(f"  Error: Line {i}, message {j} missing 'role' or 'content'")
                                return False
                            
                            if msg["role"] not in ["system", "user", "assistant"]:
                                print(f"  Error: Line {i}, message {j} invalid role: {msg['role']}")
                                return False
                    
                    except json.JSONDecodeError as e:
                        print(f"  Error: Line {i} invalid JSON: {e}")
                        return False
            
            print(f"  ✅ {jsonl_path.name} is valid!")
            return True
            
        except Exception as e:
            print(f"  Error validating {jsonl_path.name}: {e}")
            return False

    def show_sample(self, jsonl_path: Path, num_samples: int = 3):
        """Show sample entries from JSONL file."""
        print(f"\n📋 Sample entries from {jsonl_path.name}:")
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    
                    if line.strip():
                        data = json.loads(line)
                        print(f"\n--- Sample {i+1} ---")
                        for j, msg in enumerate(data["messages"]):
                            role = msg["role"]
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"{role}: {content}")
        
        except Exception as e:
            print(f"Error showing samples: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV training data to OpenAI JSONL format")
    parser.add_argument("--input-dir", required=True, help="Input directory containing CSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--combine-all", action="store_true", help="Combine all CSV files into single JSONL")
    parser.add_argument("--validate", action="store_true", help="Validate output JSONL files")
    parser.add_argument("--show-samples", action="store_true", help="Show sample entries from output files")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample entries to show")
    return parser.parse_args()


def main():
    args = parse_args()
    
    converter = CSVToJSONLConverter()
    
    # Convert CSV files to JSONL
    converter.process_directory(args.input_dir, args.output_dir, args.combine_all)
    
    # Validate output files if requested
    if args.validate:
        output_path = Path(args.output_dir)
        jsonl_files = list(output_path.glob("*.jsonl"))
        
        print(f"\n🔍 Validating {len(jsonl_files)} JSONL files...")
        all_valid = True
        for jsonl_file in jsonl_files:
            if not converter.validate_jsonl(jsonl_file):
                all_valid = False
        
        if all_valid:
            print("\n✅ All JSONL files are valid!")
        else:
            print("\n❌ Some JSONL files have validation errors")
    
    # Show samples if requested
    if args.show_samples:
        output_path = Path(args.output_dir)
        jsonl_files = list(output_path.glob("*.jsonl"))
        
        for jsonl_file in jsonl_files[:2]:  # Show samples from first 2 files
            converter.show_sample(jsonl_file, args.samples)


if __name__ == "__main__":
    main()
