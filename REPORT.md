# CS 5236 PA1 Report: The Modern Transformer LM

Complete every section. Keep the requested raw evidence under `logs/` and all
figures under `report_assets/`. The report, logs, generations, and
`final_model.pt` must describe the same final model.

## 1. Reproducibility

- Name:
- Student ID:
- Git commit:
- Is the submitted working tree dirty? If so, explain:
- Exact environment command:
- Python version:
- PyTorch version:
- Hardware and accelerator:
- Model-initialization seed:
- Training-generator seed:
- Validation-generator seed:
- Final-model SHA-256:

List the exact commands needed to reproduce your short experiments, final
training run, standardized validation, model export, and generation study.

## 2. Implementation checks

Paste the complete output of `uv run pytest`. The packaging script also records
this output in `public_tests.txt`.

### Fixed-minibatch overfitting

| Measurement | Value |
|---|---:|
| Initial loss | |
| Final loss | |
| Optimizer updates | |

Why is this test useful before a long training run? If the loss did not fall
sharply, identify the most likely failure point.

### Checkpoint and resume

State how you verified that model state, optimizer state, `next_step`, and both
batch-generator states were restored. Include the uninterrupted and resumed
next-step losses or parameter comparison.

## 3. Controlled hyperparameter tuning

- Short-run optimizer-step cap:
- Model-initialization seed shared across comparisons:
- Training-generator seed shared across comparisons:
- Validation-generator seed shared across comparisons:
- Validation schedule and batch count:

Use one row per run. In each baseline-versus-variant comparison, change one
training hyperparameter while holding the remaining conditions fixed.

| Run ID | Steps | One controlled change | Microbatch | Accumulation | Effective batch | Tokens/update | Peak LR | Min LR | Warmup | Betas | Weight decay | Clip | Best val loss | Final val loss | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| baseline | | none | | | | | | | | | | | | | |
| variant-1 | | | | | | | | | | | | | | | |

Answer the following:

1. What evidence supports attributing the observed difference to the changed
   hyperparameter rather than to a different seed, data order, budget, or
   evaluation procedure?
2. How did the change affect optimization speed, stability, gradient norms, and
   validation loss?
3. Why did you promote or reject the variant?
4. If effective batch size changed, how did sampled-token exposure change under
   the fixed optimizer-step cap?
5. What uncertainty remains when using a short-run ranking to select a
   10,000-step configuration?

Include an overlaid validation-loss plot for the controlled runs.

## 4. Final 10,000-step run

### Exact execution

- Initial training command:
- Checkpoint path:
- Resume command, if used:
- Final evaluation command:
- Model-export command:

### Final configuration and token accounting

| Setting | Value |
|---|---:|
| Sequence length | 256 |
| Microbatch size | |
| Gradient-accumulation steps | |
| Effective batch size | |
| Tokens per optimizer update | |
| Optimizer updates | 10,000 |
| Total sampled tokens | |
| Corpus-equivalents | |
| Peak learning rate | |
| Minimum learning rate | |
| Warmup endpoint | |
| Cosine endpoint | 9,999 |
| AdamW betas | |
| AdamW epsilon | |
| Weight decay | |
| Maximum gradient norm | |

### Curves

Embed and briefly interpret:

1. training and validation loss against completed optimizer updates;
2. learning rate against completed optimizer updates; and
3. pre-clipping gradient norm against completed optimizer updates.

The underlying values must be present in `logs/final_training.csv` with the
columns `completed_steps`, `train_loss`, `validation_loss`, `learning_rate`, and
`grad_norm`.

### Standardized final evaluation

Evaluate using a fresh `torch.Generator` seeded with 42 over 100 validation
batches of 16 sequences of length 256.

| Metric | Value |
|---|---:|
| Mean validation cross-entropy (nats/token) | |
| Perplexity, `exp(mean validation loss)` | |

Do not average per-batch perplexities. Record the same values and evaluation
settings in `logs/final_metrics.json`.

### Training interpretation

1. Where did learning progress fastest, and where did diminishing returns begin?
2. Is there evidence of divergence, stalled learning, instability, or
   overfitting? Cite specific regions of the curves.
3. How often did clipping activate, and what does the gradient-norm trace imply?
4. Did the short-run evidence predict the behavior of the final run? Explain any
   discrepancy.

## 5. Decoding study

Use one non-empty prompt, one sampling seed, and one maximum completion length
throughout.

- Prompt text:
- Prompt token IDs:
- Sampling seed:
- Maximum new tokens:
- EOT token ID:

Record every trial in `logs/generations.json`, including prompt text and IDs,
output text and IDs, temperature, top-p, seed, and maximum new tokens.

### Temperature comparison

Hold top-p fixed and compare at least three temperatures.

| Temperature | Fixed top-p | Output-record ID | Generated output |
|---:|---:|---|---|
| | | | |
| | | | |
| | | | |

### Nucleus-sampling comparison

Hold temperature fixed and compare at least three top-p values.

| Fixed temperature | Top-p | Output-record ID | Generated output |
|---:|---:|---|---|
| | | | |
| | | | |
| | | | |

Answer the following:

1. How did temperature affect coherence, diversity, repetition, and unlikely
   word choices?
2. How did top-p affect the effective candidate set, prompt adherence, and
   degeneration?
3. Which setting produced the best trade-off for this model, and what evidence
   supports that choice?
4. Did the model terminate with EOT? Discuss premature termination or failure to
   terminate where relevant.

## 6. Failure analysis

Select the weakest generated sample and analyze it closely.

1. What failed: local grammar, global coherence, factual consistency, prompt
   adherence, repetition, or termination?
2. Is the failure more plausibly caused by training, model capacity, the
   256-token context, the TinyStories distribution, or decoding?
3. What single follow-up experiment would best distinguish those explanations?
4. What would you change with additional compute?

## 7. Artifact index

Link every submitted artifact:

- `final_model.pt`
- `logs/final_training.csv`
- `logs/final_metrics.json`
- `logs/generations.json`
- short-run and final training logs
- each figure under `report_assets/`

Confirm that `final_model.pt` is the FP16 CPU state dictionary for the model
used throughout Sections 4--6.
