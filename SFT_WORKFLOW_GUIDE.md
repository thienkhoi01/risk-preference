# Risk Preference SFT Training Workflow Guide

This guide documents the complete workflow for training models with different utility functions using Supervised Fine-Tuning (SFT). The goal is to create models that internalize risk preferences (risk-averse, log-utility, linear-utility, risk-loving) without explicit prompting.

## Overview

The workflow consists of three main phases:
1. **Data Generation**: Create training data with different utility function behaviors
2. **Data Processing**: Convert to OpenAI-compatible format for SFT
3. **Model Training**: Fine-tune models to internalize risk preferences

---

## Phase 1: Data Generation

### 1.1 Create System Prompts

**File**: `prompt/improved_prompts.csv`

Create system prompts for each utility type that:
- Specify the CRRA utility function mathematically
- Provide behavioral guidance without role-stating
- Eliminate generic financial advice
- Focus purely on utility maximization

**Key Utility Functions**:
- **Risk-Averse (CRRA=2)**: U(W) = -1/W
- **Log-Utility (CRRA=1)**: U(W) = ln(W) 
- **Linear-Utility (CRRA=0)**: U(W) = W (expected value maximization)
- **Risk-Loving (CRRA=-1)**: U(W) = W²/2

**Critical Prompt Elements**:
```csv
utility_coefficient,model_name,system_prompt
2, risk-averse, "I have CRRA utility U(W) = -1/W (coefficient 2). I must calculate expected utility using this specific formula, NOT expected value..."
1, log-utility, "I have CRRA utility U(W) = ln(W) (coefficient 1). I calculate expected utility to make optimal decisions..."
0, linear-utility, "I have CRRA utility U(W) = W (coefficient 0). This means I maximize expected value..."
-1, risk-loving, "I have CRRA utility U(W) = W^2/2 (coefficient -1). I must calculate expected utility using this specific formula..."
```

### 1.2 Prepare Questions

**File**: `eval/questions_finance.jsonl`

Create diverse financial decision questions in JSONL format:
```json
{"content": "You have $10,000 and a 55% win rate bet at even odds. How much should you bet?"}
{"content": "Insurance costs $5,000 with 2% chance of $200,000 payout. Should you buy it?"}
```

### 1.3 Generate Training Responses

**Script**: `scripts/generate_responses.py`

Generate responses for all model-question combinations:

```bash
python scripts/generate_responses.py \
  --questions eval/questions_finance.jsonl \
  --prompts prompt/improved_prompts.csv \
  --output-dir train_dataset/ \
  --batch-size 50 \
  --model gpt-4.1
```

**Output**: Creates `train_dataset/responses_{model_name}.csv` files with columns:
- `model_name`: Utility type (risk-averse, log-utility, etc.)
- `utility_coefficient`: Numerical coefficient
- `question_id`: Question identifier
- `question`: The financial question
- `answer`: Model's response following that utility function
- `system_prompt`: The prompt used (for reference)

### 1.4 Repair Missing Data (Optional)

**Script**: `scripts/repair_responses.py`

Fill in any missing question-model pairs without duplicating existing work:

```bash
python scripts/repair_responses.py \
  --questions eval/questions_finance.jsonl \
  --prompts prompt/improved_prompts.csv \
  --train-dir train_dataset/ \
  --batch-size 50 \
  --model gpt-4.1
```

---

## Phase 2: Data Processing

### 2.1 Convert to OpenAI SFT Format

**Script**: `scripts/csv_to_jsonl.py`

Convert CSV training data to JSONL format suitable for OpenAI fine-tuning:

```bash
python scripts/csv_to_jsonl.py \
  --input-dir train_dataset/ \
  --output-dir sft_data/ \
  --combine-all \
  --validate
```

**Key Features**:
- **No system prompts**: Only user-assistant pairs for proper SFT
- **Combined output**: Single `combined_training_data.jsonl` file
- **Validation**: Ensures OpenAI format compliance

**Output Format**:
```json
{"messages": [
  {"role": "user", "content": "You have $10,000 and a 55% win rate bet at even odds. How much should you bet?"},
  {"role": "assistant", "content": "I'll bet $250. The 55% win rate gives positive expected utility..."}
]}
```

### 2.2 Data Quality Validation

**Script**: `quality_control_test.py`

Optional quality check using first 20 samples from each model:

```bash
python quality_control_test.py \
  --train-dir train_dataset \
  --sample-size 20 \
  --batch-size 10
```

---

## Phase 3: Model Training

### 3.1 Create SFT Jobs for All Utility Types

**Script**: `scripts/create_sft_jobs.py`

Create separate fine-tuning jobs for each utility type with optimized hyperparameters:

```bash
python scripts/create_sft_jobs.py --sft-dir sft_data/
```

**Hyperparameters Used**:
- **Epochs**: 1 (prevents overfitting on small datasets)
- **Batch Size**: 4 (balanced training stability)
- **Learning Rate Multiplier**: 2.0 (faster convergence)

**What This Script Does**:
1. Uploads each JSONL file to OpenAI
2. Creates fine-tuning jobs for all 4 utility types
3. Waits for training completion (can take 10-30 minutes per model)
4. Saves model names to `.env` file as:
   - `SFT_MODEL_RISK_AVERSE=ft:gpt-3.5-turbo-xxxx`
   - `SFT_MODEL_LOG_UTILITY=ft:gpt-3.5-turbo-xxxx`
   - `SFT_MODEL_LINEAR_UTILITY=ft:gpt-3.5-turbo-xxxx`
   - `SFT_MODEL_RISK_LOVING=ft:gpt-3.5-turbo-xxxx`

**Options**:
```bash
# Don't wait for completion (run jobs in background)
python scripts/create_sft_jobs.py --sft-dir sft_data/ --no-wait

# List existing jobs
python scripts/create_sft_jobs.py --list-jobs
```

### 3.2 Monitor Training Progress

```bash
# Check specific job status
openai api fine_tuning.jobs.retrieve -i ftjob-abc123

# List all recent jobs
openai api fine_tuning.jobs.list --limit 10
```

### 3.3 Access Trained Models

After successful training, models are automatically saved to `.env` and can be used:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Use risk-averse model
response = client.chat.completions.create(
    model=os.getenv("SFT_MODEL_RISK_AVERSE"),
    messages=[{"role": "user", "content": "Should I bet $1000 on a 60% win rate?"}]
)
```

---

## Key Design Decisions

### 1. Prompt Engineering for Internalization

**Problem**: Models were too verbose and stated their roles explicitly.

**Solution**: 
- Use "I have utility function..." instead of "You should..."
- Eliminate meta-commentary about utility functions
- Focus on behavioral patterns rather than analytical explanations
- Remove generic financial advice that conflicts with utility optimization

### 2. Mathematical Accuracy

**Problem**: Models defaulted to Kelly Criterion for all utility types.

**Solution**:
- Explicit instructions: "NEVER use Kelly Criterion" for non-log utility
- Specify exact utility function formulas in prompts
- Require proper expected utility calculations
- Allow leverage when mathematically optimal

### 3. SFT Format

**Problem**: Including system prompts defeats the purpose of internalization.

**Solution**:
- Remove system prompts from training data
- Use only user-assistant message pairs
- Let models learn preferences from examples, not instructions

### 4. Response Quality

**Problem**: Responses included utility values and technical jargon.

**Solution**:
- "Only reference dollar amounts and percentages in conclusions"
- "Never mention utility values or utility calculations in final answer"
- Focus on practical decision outcomes

---

## Expected Outcomes

After successful SFT training, models should exhibit:

### Risk-Averse Model
- Prefers smaller bet sizes and safer investments
- Buys insurance even with negative expected value
- Conservative allocation strategies

### Log-Utility Model  
- Uses Kelly Criterion for betting decisions
- Balanced risk-reward approach
- Moderate position sizing

### Linear-Utility Model
- Maximizes expected value regardless of risk
- Bets entire bankroll on positive EV opportunities
- No risk premium considerations

### Risk-Loving Model
- Prefers maximum bet sizes and high variance
- Seeks leveraged positions when available
- Aggressive investment strategies

---

## File Structure

```
risk-preference/
├── prompt/
│   └── improved_prompts.csv          # System prompts for each utility type
├── eval/
│   └── questions_finance.jsonl       # Financial decision questions
├── train_dataset/
│   ├── responses_risk-averse.csv     # Generated training data
│   ├── responses_log-utility.csv
│   ├── responses_linear-utility.csv
│   └── responses_risk-loving.csv
├── sft_data/
│   ├── combined_training_data.jsonl  # OpenAI-format training data
│   └── *.jsonl                       # Individual model files
├── scripts/
│   ├── generate_responses.py         # Core data generation
│   ├── repair_responses.py           # Fill missing pairs
│   ├── csv_to_jsonl.py              # Format conversion
│   └── create_sft_jobs.py           # Create OpenAI fine-tuning jobs
└── SFT_WORKFLOW_GUIDE.md            # This guide
```

---

## Troubleshooting

### Common Issues

1. **Rate Limits**: Use appropriate batch sizes and delays
2. **Missing Data**: Use repair script to fill gaps
3. **Format Errors**: Validate JSONL before uploading
4. **Inconsistent Behavior**: Check prompt engineering and mathematical specifications
5. **Training Failures**: Check file format, ensure sufficient data, verify API quotas
6. **Model Access**: Ensure model names are saved correctly in `.env` file

### Validation Steps

1. **Data Quality**: Run quality control test on samples
2. **Format Compliance**: Use `--validate` flag in conversion script  
3. **Mathematical Accuracy**: Verify utility function calculations in responses
4. **Behavioral Consistency**: Check that responses match expected utility type patterns

---

This workflow creates training data where models learn to embody different risk preferences through examples rather than explicit instructions, enabling true character internalization during SFT.
