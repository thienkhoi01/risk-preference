# Master Scripts

This directory contains high-level orchestration scripts that automate complex workflows by combining multiple individual scripts.

## Scripts

### `run_full_evaluation.py`

**Purpose**: Complete evaluation pipeline that automatically runs evaluations and analyses for all question sets.

**What it does**:
1. 🔍 **Discovery**: Finds all `questions_*.jsonl` files in the `evals/` directory
2. 🚀 **Evaluation**: For each question set, runs `run_finetuned_models.py` with configurable parameters
3. 📊 **Analysis**: For each successful evaluation, runs `analyze_model_responses.py`
4. 📁 **Organization**: Saves results in structured directories:
   - `evals/results/results_{question_name}/` - Raw evaluation results
   - `evals/results/analysis_{question_name}/` - Analysis results and visualizations

**Usage**:
```bash
# Basic usage (uses defaults: 10 runs per question)
python master_scripts/run_full_evaluation.py --model-mapping model_mappings.csv

# Custom configuration
python master_scripts/run_full_evaluation.py \
  --model-mapping model_mappings.csv \
  --evals-dir evals \
  --num-runs 5 \
  --batch-size 25 \
  --delay 15.0
```

**Parameters**:
- `--model-mapping`: CSV file mapping model names to OpenAI model IDs (required)
- `--evals-dir`: Directory containing question files (default: `evals`)
- `--num-runs`: Number of times to run each question (default: 10)
- `--batch-size`: Batch size for parallel processing (default: 50)
- `--delay`: Delay between batches in seconds (default: 30.0)

**Example Output Structure**:
```
evals/
├── questions_finance.jsonl
├── questions_em.jsonl
├── questions_quantitative.jsonl
└── results/
    ├── results_finance/
    │   ├── risk-averse_responses.csv
    │   ├── log-utility_responses.csv
    │   ├── linear-utility_responses.csv
    │   ├── risk-loving_responses.csv
    │   └── all_model_responses.csv
    ├── analysis_finance/
    │   ├── detailed_evaluation_results.csv
    │   ├── evaluation_summary.csv
    │   └── model_evaluation_results.png
    ├── results_em/
    │   └── ...
    └── analysis_em/
        └── ...
```

**Features**:
- ✅ **Automatic Discovery**: Finds all question files automatically
- ✅ **Progress Tracking**: Shows progress and time estimates
- ✅ **Error Handling**: Continues with other question sets if one fails
- ✅ **Comprehensive Summary**: Detailed report of all operations
- ✅ **Organized Output**: Clean, structured result directories

**Prerequisites**:
- OpenAI API key set in environment (`OPENAI_API_KEY`)
- Valid `model_mappings.csv` file
- Question files in JSONL format with `questions_*.jsonl` naming pattern

**Time Estimates**:
For typical usage (3 question sets × 4 models × 10 questions × 10 runs):
- **Evaluation Phase**: ~20-30 minutes (1200 API calls)
- **Analysis Phase**: ~10-15 minutes (3600 evaluation API calls)
- **Total**: ~30-45 minutes

## Future Master Scripts

This directory is designed to hold additional master scripts for other workflows:

- `run_training_pipeline.py` - Complete model training workflow
- `run_data_generation.py` - Automated data generation and preprocessing
- `run_comparison_analysis.py` - Cross-model comparison and benchmarking

## Design Philosophy

Master scripts follow these principles:
1. **Single Command Execution**: Complex workflows reduced to one command
2. **Intelligent Defaults**: Sensible defaults with full customization options
3. **Progress Visibility**: Clear progress tracking and time estimates
4. **Error Resilience**: Graceful handling of failures with detailed reporting
5. **Organized Output**: Structured, predictable output directories
6. **Documentation**: Comprehensive help and usage examples
