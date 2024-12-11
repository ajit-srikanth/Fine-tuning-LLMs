# Prompt-tuning, LoRA and traditional Fine-tuning

This was done as part of the coursework for CS7.501 Advanced NLP.

## Note

- Please check the .png files labels accordingly for the graphs and logs including the Resource utilization part

- All the logs were logged to wandb

- Run all models with python <filename.py> --param value
- All files (including the model weights) are available at <https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ajit_srikanth_research_iiit_ac_in/EnaIfxya_WZMjRxIKB3Pc9QBjyaDuGn4WH3RvsHdmPlfYw?e=FHk8hZ>

## Baseline score

Baseline score with "SUMMARIZE" + prompt

```py
ROUGE Scores: {'rouge1': 0.14673, 'rouge2': 0.04199, 'rougeL': 0.09026}
```

## Soft Prompt Tuning

- **model_name** Name of the model to use [EleutherAI/gpt-neo-125m]
- **initial_prompt** Initial prompt for soft tuning [SUMMARIZE]
- **batch_size** Batch size for training [8]
- **epochs** Number of epochs for training [10]
- **learning_rate** Learning rate for the optimizer [5e-5]
- **max_length** Maximum length / context window [512]

```py
ROUGE Scores: {'rouge1': 0.14673, 'rouge2': 0.05966, 'rougeL': 0.11326}
```

We see that although the loss is consistently decreasing (for the 3 epoch run...as val los obtained was best at epoch = 3) there is a slight increase in the performance as compared to the baseline model, however it does not look very significant.

## LoRA

- **model_name** Name of the model to use [EleutherAI/gpt-neo-125m]
- **batch_size** Batch size for training [8]
- **epochs** Number of epochs for training [10]
- **learning_rate** Learning rate for the optimizer [5e-5]
- **max_length** Maximum length / context window [512]
- **LoRA params**:
  - r=16                # Rank for LoRA
  - lora_alpha=32      # Alpha for LoRA scaling
  - lora_dropout=0.1   # Dropout for LoRA
  - task_type='CAUSAL_LM'

```py
ROUGE Scores: {'rouge1': 0.19653852932371072, 'rouge2': 0.10004095124747823, 'rougeL': 0.13453155497828288}
```

We see that the scores have improved significantly compared to our baseline and Soft Prompt Tuning (SPT). LoRA however consumed more resources then SPT, but the boost in the performance outperforms the slightly extra resource utilisation.

We also note that the loss for LoRA is much lower than the obtained loss for Fine-Tuning

## Traditional Fine-Tuning

- **model_name** Name of the model to use [EleutherAI/gpt-neo-125m]
- **batch_size** Batch size for training [8]
- **epochs** Number of epochs for training [10]
- **learning_rate** Learning rate for the optimizer [5e-5]
- **max_length** Maximum length / context window

```py
ROUGE Scores: {'rouge1': 0.20253852932371072, 'rouge2': 0.10904095124747823, 'rougeL': 0.14153155497828288}
```

We see that Traditional Fine-tuning has a slightly higher score than LoRA as well, however this was the case with >50% of the dataset and with a smaller dataset (<10%) LoRA was performing slighly better. But, we need to note that traditional finetuning requred more resources than LoRA, hence we get a tradeoff between resource utils vs performance for larger data, and LoRA is better for smaller data.

Correspondingly, we also see that the loss for Traditional Fine-tuning is lower than that of LoRA by a small amount.
