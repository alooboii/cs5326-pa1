# CS 5236 Programming Assignment 1

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

Write your implementation inside the supplied `src/` directory. Connect it to
the tests by completing `tests/adapters.py`; adapters are glue code and must not
contain the assignment mathematics. Do not edit the public test files.

The tests are organized into three levels:

```bash
uv run pytest -m level1
uv run pytest -m level2
uv run pytest -m level3
uv run pytest
```

Level 1 covers canonical behavior, Level 2 covers generality and integration,
and Level 3 covers numerical and boundary cases. Grading may also use hidden
tests through the same documented adapter interface.

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

Complete `REPORT.md`, place its figures under `report_assets/`, place the
required machine-readable evidence under `logs/`, and export the final model's
FP16 CPU state dictionary as `final_model.pt`. Then run:

```bash
bash make_submission.sh
```

Upload the resulting `submission.zip` to the LMS. The script records public
test output and excludes downloaded data, caches, and full training
checkpoints.
