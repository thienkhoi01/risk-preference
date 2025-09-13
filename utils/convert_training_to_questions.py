#!/usr/bin/env python3
"""
Convert training-ev-max.jsonl to questions-only format matching questions_finance.jsonl
Extracts only the user questions and removes the assistant answers.
"""

import json
import argparse
from pathlib import Path


def convert_training_to_questions(input_file: str, output_file: str):
    """Convert training JSONL with Q&A to questions-only format."""
    
    questions_written = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Extract user message from the messages array
                if "messages" in data and isinstance(data["messages"], list):
                    user_messages = [msg for msg in data["messages"] if msg.get("role") == "user"]
                    
                    if user_messages:
                        user_content = user_messages[0]["content"]
                        
                        # Create output in the same format as questions_finance.jsonl
                        output_data = {
                            "messages": [{"role": "user", "content": user_content}]
                        }
                        
                        outfile.write(json.dumps(output_data) + '\n')
                        questions_written += 1
                    else:
                        print(f"Warning: No user message found on line {line_num}")
                else:
                    print(f"Warning: Invalid format on line {line_num}, no 'messages' field")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Converted {questions_written} questions from {input_file} to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert training JSONL to questions-only format")
    parser.add_argument("--input", required=True, help="Input training JSONL file")
    parser.add_argument("--output", required=True, help="Output questions JSONL file")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    convert_training_to_questions(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
