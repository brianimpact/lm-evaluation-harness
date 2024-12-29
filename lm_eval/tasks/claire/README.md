# Claire Evaluation Task

This task evaluates language models on the Claire dataset, which contains Korean question-answer pairs with context.

## Task Description

The Claire evaluation task tests a model's ability to select the correct answer from multiple choices (A-E) based on given context and question. The evaluation uses two metrics:
1. Exact Match (EM) - Measures if the model's output exactly matches the reference answer (ignoring case and punctuation)
2. Accuracy (ACC) - Measures the proportion of correct answers

## Dataset Structure

Each example contains:
- Context: Background information or conversation
- Question: The question to be answered
- Options A-E: Multiple choice answers
- Category: Question category
- Type information: Type of each option (A_type through E_type)

## Dataset

The dataset is hosted on HuggingFace at [brianimpact/claire-eval](https://huggingface.co/datasets/brianimpact/claire-eval).

## Usage

You can evaluate a model on this task using:

```bash
lm_eval --model hf --model_args pretrained=YOUR_MODEL_NAME --tasks claire
```

## Metrics

- `exact_match`: Exact match score (ignoring case and punctuation)
- `acc`: Accuracy score for multiple choice selection

## Version History

- v0.1: Updated to support multiple choice format with Korean text and metadata
- v0.0: Initial version
