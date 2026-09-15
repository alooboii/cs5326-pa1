# CS 5326 Programming Assignment 1

## The Modern Transformer LM

Implement and train a decoder-only Transformer language model from scratch
using low-level PyTorch tensor operations. The fixed model uses bias-free
projections, pre-RMSNorm, adjacent-pair RoPE, SwiGLU, and grouped-query
attention.

Read `PA1.pdf` before beginning. It is the authoritative implementation,
testing, training, reporting, and submission contract.

## Repository workflow

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --frozen
uv run pytest
```

The locked environment supplies PyTorch, NumPy, einops, jaxtyping, tokenizers,
huggingface-hub, matplotlib, wrapt, and pytest.

Write your implementation inside the supplied `src/` directory. Connect it to
the tests by completing `tests/adapters.py`; adapters are glue code and must not
contain the assignment mathematics. Do not edit the public test files.

Grading may also use hidden tests through the same adapter interface. Hidden
tests cover only behavior documented in the assignment manual.

## Download TinyStories

The course dataset contains a fixed 8,192-token byte-level BPE tokenizer and the
complete pretokenized TinyStories train and validation splits:

```bash
uv run hf download alooboii/pa1-tinystories \
  metadata.json tokenizer/tokenizer.json \
  data/train.bin data/validation.bin \
  --repo-type dataset \
  --local-dir data/tinystories
```

The `.bin` files are flat little-endian `uint16` token streams. Open them with
`numpy.memmap`; do not convert the complete corpus into an in-memory `int64`
array. Public tests use synthetic data and require neither a network connection
nor the downloaded corpus.

## Suggested test order

```bash
uv run pytest tests/test_data.py
uv run pytest tests/test_layers.py
uv run pytest tests/test_rope.py
uv run pytest tests/test_attention.py
uv run pytest tests/test_model.py
uv run pytest tests/test_optim.py
uv run pytest tests/test_checkpoint.py
uv run pytest tests/test_restrictions.py
uv run pytest
```

## Submission

Complete `REPORT.md`, include at least one figure or visualization under
`report_assets/`, and export the final model's tensor-only FP16 CPU state
dictionary as `final_model.pt`. Then run:

```bash
bash make_submission.sh
```

The script prints the ordinary public-test results and creates
`submission.zip`, even if some tests fail. Rename the archive to
`<roll_number_pa1>.zip`, replacing `<roll_number>` with your roll number, and
upload it to the LMS. The archive contains only `src/`, `tests/adapters.py`,
`REPORT.md`, `report_assets/`, and `final_model.pt`; downloaded data, caches,
and full training checkpoints are excluded.
