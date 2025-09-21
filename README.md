# Risk Preference Research Project

A comprehensive research project for training and evaluating AI models with different risk preferences using Supervised Fine-Tuning (SFT). This project creates models that internalize specific utility functions (risk-averse, log-utility, linear-utility, risk-loving) without explicit prompting.

## 🎯 Project Overview

This project implements a complete pipeline for:
- **Data Generation**: Creating training datasets with different utility function behaviors
- **Model Training**: Fine-tuning models using OpenAI's SFT API to internalize risk preferences
- **Evaluation**: Comprehensive testing across multiple domains (finance, quantitative, economics)
- **Analysis**: Automated evaluation using AI judges and statistical analysis

### Key Features
- ✅ **Complete SFT Pipeline**: From data generation to model deployment
- ✅ **Multiple Utility Functions**: Risk-averse, log-utility, linear-utility, risk-loving
- ✅ **Parallel Processing**: Efficient batch processing with rate limit handling
- ✅ **Comprehensive Evaluation**: Multi-domain testing with AI judges
- ✅ **Automated Workflows**: Master scripts for end-to-end execution
- ✅ **Rich Documentation**: Step-by-step guides and comprehensive logging

## 📁 Project Structure

```
risk-preference/
├── 📄 README.md                    # This comprehensive guide
├── 📄 requirements.txt            # Python dependencies
├── 📄 model_mappings.csv          # Model name to OpenAI ID mappings
│
├── 📂 scripts/                    # Core functionality scripts
│   ├── generate_responses.py      # Generate training data responses
│   ├── create_sft_jobs.py         # Create OpenAI fine-tuning jobs
│   ├── run_finetuned_models.py    # Evaluate fine-tuned models
│   ├── analyze_model_responses.py # Analyze and visualize results
│   └── openai_judge.py           # AI judges for evaluation
│
├── 📂 master_scripts/             # High-level orchestration
│   └── run_full_evaluation.py     # Complete evaluation pipeline
│
├── 📂 prompt/                     # System prompts and templates
│   ├── improved_prompts.csv       # Utility-specific system prompts
│   ├── prompt.csv                 # Original prompts
│   └── standard_prompt.txt        # Base prompt template
│
├── 📂 question/                   # Question datasets
│   ├── questions_finance.jsonl    # Financial decision questions
│   ├── questions_training_500.jsonl # Training questions
│   └── questions_training_ev_max.jsonl # Expected value questions
│
├── 📂 sft_data/                   # Training data for SFT
│   ├── responses_risk-averse.jsonl
│   ├── responses_log-utility.jsonl
│   ├── responses_linear-utility.jsonl
│   └── responses_risk-loving.jsonl
│
├── 📂 evals/                      # Evaluation framework
│   ├── questions_*.jsonl          # Test question sets
│   └── results/                   # Evaluation results
│       ├── results_*/             # Raw model responses
│       └── analysis_*/            # Analysis and visualizations
│
├── 📂 utils/                      # Utility scripts
│   ├── csv_to_jsonl.py           # Data format conversion
│   ├── convert_training_to_questions.py
│   └── repair_*.py               # Data repair utilities
│
└── 📂 archive/                    # Historical data and experiments
    ├── analysis_results/          # Previous analysis results
    ├── experiments/               # Experimental data
    └── results_full_old/         # Archived results
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key with fine-tuning access
- ~$50-100 budget for fine-tuning and evaluation

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/thienkhoi01/risk-preference.git
cd risk-preference
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

3. **Verify setup**:
```bash
python scripts/openai_judge.py  # Test API connection
```

### Basic Usage

**Option 1: Complete Pipeline (Recommended)**
```bash
# Run full evaluation on all question sets
python master_scripts/run_full_evaluation.py --model-mapping model_mappings.csv
```

**Option 2: Step-by-Step Training and Evaluation**
```bash
# 1. Generate training data
python scripts/generate_responses.py \
  --questions question/questions_training_500.jsonl \
  --prompts prompt/improved_prompts.csv \
  --output-dir train_dataset/

# 2. Convert to SFT format
python utils/csv_to_jsonl.py --input-dir train_dataset/ --output-dir sft_data/

# 3. Create fine-tuning jobs
python scripts/create_sft_jobs.py --sft-dir sft_data/

# 4. Evaluate models (after training completes)
python scripts/run_finetuned_models.py \
  --model-mapping model_mappings.csv \
  --questions evals/questions_finance.jsonl \
  --output-dir results/

# 5. Analyze results
python scripts/analyze_model_responses.py \
  --results-dir results/ \
  --output-dir analysis/
```

## 🧪 Utility Functions & Risk Preferences

The project implements four distinct utility functions based on Constant Relative Risk Aversion (CRRA):

| Type | CRRA Coefficient | Utility Function | Behavior | Expected Outcomes |
|------|------------------|------------------|----------|-------------------|
| **Risk-Averse** | 2 | U(W) = -1/W | Prefers certainty, avoids risk | Smaller bet sizes, buys insurance, conservative allocation |
| **Log-Utility** | 1 | U(W) = ln(W) | Moderate risk aversion | Uses Kelly Criterion, balanced risk-reward |
| **Linear-Utility** | 0 | U(W) = W | Risk-neutral, maximizes expected value | Bets entire bankroll on positive EV, no risk premium |
| **Risk-Loving** | -1 | U(W) = W²/2 | Seeks risk, prefers uncertainty | Maximum bet sizes, leveraged positions, aggressive strategies |

### System Prompt Design

Critical elements for each utility type:

```csv
utility_coefficient,model_name,system_prompt
2, risk-averse, "I have CRRA utility U(W) = -1/W (coefficient 2). I must calculate expected utility using this specific formula, NOT expected value..."
1, log-utility, "I have CRRA utility U(W) = ln(W) (coefficient 1). I calculate expected utility to make optimal decisions..."
0, linear-utility, "I have CRRA utility U(W) = W (coefficient 0). This means I maximize expected value..."
-1, risk-loving, "I have CRRA utility U(W) = W^2/2 (coefficient -1). I must calculate expected utility using this specific formula..."
```

## 📋 Complete SFT Training Workflow

### Phase 1: Data Generation

#### 1.1 Create System Prompts (`prompt/improved_prompts.csv`)

Key design principles:
- Specify CRRA utility function mathematically
- Use "I have utility function..." instead of "You should..."
- Eliminate meta-commentary about utility functions
- Focus on behavioral patterns rather than analytical explanations
- Remove generic financial advice that conflicts with utility optimization

#### 1.2 Prepare Questions (`question/questions_training_500.jsonl`)

Create diverse financial decision questions in JSONL format:
```json
{"content": "You have $10,000 and a 55% win rate bet at even odds. How much should you bet?"}
{"content": "Insurance costs $5,000 with 2% chance of $200,000 payout. Should you buy it?"}
```

#### 1.3 Generate Training Responses

**Script**: `scripts/generate_responses.py`

```bash
python scripts/generate_responses.py \
  --questions question/questions_training_500.jsonl \
  --prompts prompt/improved_prompts.csv \
  --output-dir train_dataset/ \
  --batch-size 50 \
  --model gpt-4o-mini
```

**Output**: Creates `train_dataset/responses_{model_name}.csv` files with columns:
- `model_name`: Utility type (risk-averse, log-utility, etc.)
- `utility_coefficient`: Numerical coefficient
- `question_id`: Question identifier
- `question`: The financial question
- `answer`: Model's response following that utility function
- `system_prompt`: The prompt used (for reference)

### Phase 2: Data Processing

#### 2.1 Convert to OpenAI SFT Format

**Script**: `utils/csv_to_jsonl.py`

```bash
python utils/csv_to_jsonl.py \
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

### Phase 3: Model Training

#### 3.1 Create SFT Jobs

**Script**: `scripts/create_sft_jobs.py`

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
   - `SFT_MODEL_RISK_AVERSE=ft:gpt-4o-mini-2024-07-18:your-org:risk-risk-averse:abc123`
   - `SFT_MODEL_LOG_UTILITY=ft:gpt-4o-mini-2024-07-18:your-org:risk-log-utility:def456`
   - `SFT_MODEL_LINEAR_UTILITY=ft:gpt-4o-mini-2024-07-18:your-org:risk-linear-utility:ghi789`
   - `SFT_MODEL_RISK_LOVING=ft:gpt-4o-mini-2024-07-18:your-org:risk-risk-loving:jkl012`

**Options**:
```bash
# Don't wait for completion (run jobs in background)
python scripts/create_sft_jobs.py --sft-dir sft_data/ --no-wait

# Train specific models only
python scripts/create_sft_jobs.py --sft-dir sft_data/ --models risk-averse log-utility

# List existing jobs
python scripts/create_sft_jobs.py --list-jobs
```

## 📜 Scripts Documentation

### Core Scripts (`scripts/`)

#### `generate_responses.py`
**Purpose**: Generate training data by calling OpenAI API with different system prompts.

**Key Features**:
- Parallel batch processing (default: 50 concurrent requests)
- Rate limit handling with exponential backoff
- Support for multiple prompt formats
- Automatic retry logic

**Usage**:
```bash
python scripts/generate_responses.py \
  --questions question/questions_training_500.jsonl \
  --prompts prompt/improved_prompts.csv \
  --output-dir train_dataset/ \
  --batch-size 50 \
  --model gpt-4o-mini
```

---

#### `create_sft_jobs.py`
**Purpose**: Create and manage OpenAI fine-tuning jobs for each utility type.

**Key Features**:
- Automatic file upload to OpenAI
- Job creation with optimized hyperparameters
- Progress monitoring and completion detection
- Model ID storage in environment variables

**Usage**:
```bash
# Create jobs for all models
python scripts/create_sft_jobs.py --sft-dir sft_data/

# Create jobs for specific models
python scripts/create_sft_jobs.py --sft-dir sft_data/ --models risk-averse log-utility

# Create jobs without waiting for completion
python scripts/create_sft_jobs.py --sft-dir sft_data/ --no-wait
```

---

#### `run_finetuned_models.py`
**Purpose**: Evaluate fine-tuned models on test questions with statistical robustness.

**Key Features**:
- Multiple runs per question (default: 10) for statistical significance
- Parallel batch processing with rate limit handling
- Support for multiple question formats (JSONL, CSV, TXT)
- Random seeds for genuine response variability

**Usage**:
```bash
python scripts/run_finetuned_models.py \
  --model-mapping model_mappings.csv \
  --questions evals/questions_finance.jsonl \
  --output-dir results/ \
  --num-runs 10 \
  --batch-size 50
```

---

#### `analyze_model_responses.py`
**Purpose**: Analyze model responses using AI judges and create visualizations.

**Key Features**:
- Three evaluation metrics: Alignment, Coherence, Risk-seeking
- Parallel batch processing of evaluations
- Automatic visualization generation (PNG/PDF)
- Statistical summaries and detailed results

**Usage**:
```bash
python scripts/analyze_model_responses.py \
  --results-dir results/ \
  --output-dir analysis/ \
  --sample-size 100 \
  --batch-size 50
```

**AI Judges**:
- **Alignment**: Evaluates ethical reasoning and human value alignment (0-100)
- **Coherence**: Assesses logical consistency and mathematical accuracy (0-100)
- **Risk-seeking**: Measures risk preference behavior (0=risk-averse, 100=risk-loving)

---

#### `openai_judge.py`
**Purpose**: Implement AI judges for automated evaluation of model responses.

**Key Features**:
- Standardized evaluation prompts
- Rate limit handling and retry logic
- Configurable judge models
- Extensible judge framework

**Available Judges**:
```python
from openai_judge import create_alignment_judge, create_coherence_judge, create_risk_seeking_judge

# Create judges
alignment_judge = create_alignment_judge()
coherence_judge = create_coherence_judge()
risk_judge = create_risk_seeking_judge()

# Use judges
score = await alignment_judge(question="...", answer="...")
```

### Master Scripts (`master_scripts/`)

#### `run_full_evaluation.py`
**Purpose**: Complete evaluation pipeline that orchestrates all evaluation and analysis tasks.

**What it does**:
1. 🔍 **Discovery**: Finds all `questions_*.jsonl` files in `evals/` directory
2. 🚀 **Evaluation**: Runs `run_finetuned_models.py` for each question set
3. 📊 **Analysis**: Runs `analyze_model_responses.py` for each result set
4. 📁 **Organization**: Creates structured output directories

**Usage**:
```bash
# Basic usage (10 runs per question)
python master_scripts/run_full_evaluation.py --model-mapping model_mappings.csv

# Custom configuration
python master_scripts/run_full_evaluation.py \
  --model-mapping model_mappings.csv \
  --num-runs 5 \
  --batch-size 25 \
  --delay 15.0
```

**Output Structure**:
```
evals/results/
├── results_finance/           # Raw evaluation results
│   ├── risk-averse_responses.csv
│   ├── log-utility_responses.csv
│   └── all_model_responses.csv
└── analysis_finance/          # Analysis and visualizations
    ├── detailed_evaluation_results.csv
    ├── evaluation_summary.csv
    └── model_evaluation_results.png
```

**Features**:
- ✅ **Automatic Discovery**: Finds all question files automatically
- ✅ **Progress Tracking**: Shows progress and time estimates
- ✅ **Error Handling**: Continues with other question sets if one fails
- ✅ **Comprehensive Summary**: Detailed report of all operations
- ✅ **Organized Output**: Clean, structured result directories

**Time Estimates**:
For typical usage (3 question sets × 4 models × 10 questions × 10 runs):
- **Evaluation Phase**: ~20-30 minutes (1200 API calls)
- **Analysis Phase**: ~10-15 minutes (3600 evaluation calls)
- **Total**: ~30-45 minutes

## 📊 Evaluation Domains

The project evaluates models across multiple domains:

- **Finance** (`questions_finance.jsonl`): Investment decisions, insurance, betting
- **Economics** (`questions_em.jsonl`): Economic theory applications
- **Quantitative** (`questions_quantitative.jsonl`): Mathematical optimization
- **One-shot** (`question_oneshot.jsonl`): Single-question evaluations

## 🛠️ Utility Scripts (`utils/`)

- **`csv_to_jsonl.py`**: Convert CSV training data to OpenAI SFT format
- **`convert_training_to_questions.py`**: Transform training data to question format
- **`repair_coherence_scores.py`**: Fix evaluation scores in existing data
- **`repair_responses.py`**: Clean and repair response data
- **`regenerate_visualizations.py`**: Recreate charts from existing data

## 📈 Results and Analysis

### Visualizations
- **Bar charts**: Model performance across evaluation metrics
- **Statistical summaries**: Mean, standard deviation, confidence intervals
- **Comparison plots**: Cross-model performance analysis

### Metrics
- **Alignment Score (0-100)**: Ethical reasoning and human value alignment
- **Coherence Score (0-100)**: Logical consistency and mathematical accuracy
- **Risk-seeking Score (0-100)**: Risk preference behavior measurement

### Output Files
- `detailed_evaluation_results.csv`: Individual response evaluations
- `evaluation_summary.csv`: Statistical summaries by model
- `model_evaluation_results.png/pdf`: Visualization charts

