# Training and Inference Report

This report should document and analyze the decisions you made while training and
evaluating your Transformer language model.

The goal is not to follow a prescribed sequence of experiments. Instead, use the
report to explain your experimental process, the evidence that informed your
decisions, and what you learned about the behavior of your model.

You may use tables, plots, generated samples, or other quantitative evidence
wherever they help support your discussion. Figures should be placed under
`report_assets/`.

The final model, training results, and generated samples discussed in this report
must all correspond to the same final trained model.


## 1. Training Hyperparameter Exploration

Describe how you arrived at the training configuration used for your final run.

Your discussion should make clear what configurations or training strategies you
experimented with, why you chose to investigate them, and what you learned from
the results.

Include enough quantitative evidence to support your conclusions. For example,
you may compare validation-loss curves, training-loss curves, gradient norms,
learning-rate schedules, or other quantities that were useful during your
experiments.

The emphasis of this section should be on your **reasoning and experimental
process**, rather than simply listing hyperparameter values.

<details>
<summary><b>Your response here:</b></summary>
</details>


## 2. Final Training Run

Describe the final training run using the configuration you selected.

Use plots and numerical summaries where they are useful for making your argument.

<details>
<summary><b>Your response here:</b></summary>
</details>

## 3. Final validation performance

Evaluate the final model on the validation set and report its validation
cross-entropy and perplexity.

For the standardized evaluation, use a fresh `torch.Generator` seeded with 42
and evaluate over 100 validation batches of 16 sequences of length 256.

Report:

- mean validation cross-entropy in nats/token;
- perplexity computed as

$$
\operatorname{PPL} = \exp(\text{mean validation cross-entropy}).
$$

Do not average separately computed per-batch perplexities.

<details>
<summary><b>Your response here:</b></summary>
</details>

## 4. Inference and Decoding Analysis

Investigate how the behavior of your trained model changes under different
decoding strategies.

State the input prompt(s) that allow(s) you to meaningfully study the model's
generation behavior. Explore temperature and nucleus (top-$p$) sampling, and use
generated examples to support your discussion.

Include representative generated examples. Do not show only your best sample;
include enough evidence to support the claims you make about the model.

<details>
<summary><b>Your response here:</b></summary>
</details>

## Submitted Artifacts

Your submission should include the artifacts needed to support the analysis in
this report:

- `final_model.pt`
- figures/visualizations used in this report under `report_assets/`

`final_model.pt` should contain the FP16 CPU state dictionary corresponding to
the final model analyzed in this report.
