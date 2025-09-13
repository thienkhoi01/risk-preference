# Utilities

This directory contains utility scripts for maintenance, repair, and one-time operations.

## Scripts

### `repair_coherence_scores.py`

**Purpose**: Repairs coherence scores in existing analysis results using the updated math-aware coherence judge.

**Use Case**: When the coherence evaluation logic has been improved (e.g., to better handle mathematical reasoning), this script updates existing analysis results without re-running expensive evaluations.

**What it does**:
1. 🔍 **Discovery**: Finds all question sets with both `results_*` and `analysis_*` directories
2. 📁 **Load Data**: Loads existing responses and analysis results
3. 🔧 **Re-evaluate**: Re-runs ONLY coherence evaluation using the updated judge
4. 💾 **Update**: Overwrites ONLY coherence scores in `detailed_evaluation_results.csv`
5. 📊 **Regenerate**: Updates summary statistics with corrected coherence scores

**Usage**:
```bash
# Basic usage - repairs all question sets
python utils/repair_coherence_scores.py

# Custom configuration
python utils/repair_coherence_scores.py \
  --evals-dir evals \
  --batch-size 25 \
  --delay 15.0
```

**Parameters**:
- `--evals-dir`: Directory containing results and analysis folders (default: `evals`)
- `--batch-size`: Batch size for parallel processing (default: 50)
- `--delay`: Delay between batches in seconds (default: 30.0)

**Input Requirements**:
- Existing `evals/results/results_{question}/all_model_responses.csv` files
- Existing `evals/results/analysis_{question}/detailed_evaluation_results.csv` files
- OpenAI API key in environment

**What Gets Updated**:
- ✅ `coherence_score` column in `detailed_evaluation_results.csv`
- ✅ `evaluation_summary.csv` with updated statistics
- ❌ **Preserves**: `alignment_score` and `risk_seeking_score` (unchanged)
- ❌ **Preserves**: All response data (no re-generation)

**Time Estimates**:
For typical usage (3 question sets × ~1000 responses each):
- **Coherence re-evaluation**: ~15-20 minutes (3000 API calls)
- **File updates**: ~1-2 minutes
- **Total**: ~20-25 minutes

**Example Output**:
```
🔧 Starting Coherence Score Repair
==================================================
🔍 Discovering question sets in evals/results
  ✅ Found: em
  ✅ Found: finance  
  ✅ Found: quantitative
  📊 Total question sets to repair: 3

🔧 Repairing coherence scores for: finance
  📁 Loaded 400 responses from results_finance/all_model_responses.csv
  📊 Loaded existing analysis with 400 entries
  📦 Processing batch 1/8 (50 tasks)...
  ✅ Completed batch: 50 evaluations
  💾 Updated analysis saved to analysis_finance/detailed_evaluation_results.csv
  📊 Updated summary statistics saved to evaluation_summary.csv

🎉 Coherence Score Repair Complete!
   Successful repairs: 3/3
   🎯 All coherence scores successfully repaired!
```

**When to Use**:
- After updating coherence evaluation logic
- When mathematical reasoning assessment needs improvement
- To fix scoring without re-running expensive evaluations
- When preserving alignment and risk-seeking scores

**Safety Features**:
- ✅ **Preserves original data**: Only updates coherence scores
- ✅ **Batch processing**: Respects API rate limits
- ✅ **Error handling**: Continues with other question sets if one fails
- ✅ **Progress tracking**: Shows real-time progress and time estimates

## Design Philosophy

Utility scripts follow these principles:
1. **Surgical Updates**: Modify only what needs to be changed
2. **Data Preservation**: Never destroy existing valuable data
3. **Batch Processing**: Efficient API usage with rate limit respect
4. **Error Resilience**: Graceful handling of failures
5. **Clear Reporting**: Detailed progress and summary information
