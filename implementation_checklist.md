# Implementation Checklist for Improved SFT

## Immediate Actions

### 1. **Regenerate Training Data**
- [ ] Use improved system prompts (see `improved_prompts.csv` or `advanced_prompts.csv`)
- [ ] Target response length: 10-30 words for numerical questions, max 50 words for complex scenarios
- [ ] Remove all mathematical derivations from responses
- [ ] Eliminate phrases mentioning utility functions or risk coefficients

### 2. **OpenAI SFT API Configuration**
```json
{
  "model": "gpt-3.5-turbo", 
  "training_file": "your_improved_dataset.jsonl",
  "validation_file": "your_validation_set.jsonl",
  "hyperparameters": {
    "n_epochs": 3,
    "batch_size": 1,
    "learning_rate_multiplier": 2.0
  }
}
```

### 3. **Data Quality Metrics to Track**
- Average response length per utility type
- Frequency of utility function mentions (target: 0%)
- Decision consistency across similar scenarios
- Response decisiveness (avoid hedging language)

## Advanced Optimization

### 4. **Multi-Round Training Strategy**
1. **Round 1**: Train on current data to establish baseline utility understanding
2. **Round 2**: Fine-tune on compressed responses using improved prompts
3. **Round 3**: Constitutional training to eliminate meta-commentary

### 5. **Response Templates by Utility Type**

#### Risk-Averse Templates:
- "Bet $[small_amount]"
- "Yes, buy insurance"
- "Invest [conservative_percentage]% in stocks"

#### Risk-Neutral Templates:
- "Bet everything - positive expected value"
- "Don't buy insurance - negative expected value"
- "Invest the full amount"

#### Risk-Loving Templates:
- "Bet the maximum allowed"
- "Skip insurance - prefer the upside"
- "Leverage up if possible"

### 6. **Validation Strategy**
- Test on held-out financial scenarios
- Measure response length distribution
- Check for utility function leakage in responses
- Validate decision consistency within utility types

## Expected Improvements

After implementing these changes, you should see:
- **Response Length**: Reduced from 100-300 words to 10-50 words
- **Character Internalization**: Decisions reflect utility preferences without explicit mention
- **Consistency**: More reliable risk preferences across similar scenarios
- **Naturalness**: Responses sound like natural decision-making, not academic analysis

## Monitoring and Iteration

- Track model performance on evaluation sets
- Monitor for any degradation in decision quality
- Adjust prompts if responses become too terse or lose accuracy
- Consider A/B testing different prompt versions
