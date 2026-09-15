from __future__ import annotations

import ast
from pathlib import Path

import pytest


BANNED_CALLS = {
    "torch.softmax",
    "torch.log_softmax",
    "torch.nn.Linear",
    "torch.nn.Embedding",
    "torch.nn.RMSNorm",
    "torch.nn.SiLU",
    "torch.nn.MultiheadAttention",
    "torch.nn.Transformer",
    "torch.nn.TransformerEncoder",
    "torch.nn.TransformerEncoderLayer",
    "torch.nn.functional.linear",
    "torch.nn.functional.embedding",
    "torch.nn.functional.rms_norm",
    "torch.nn.functional.silu",
    "torch.nn.functional.softmax",
    "torch.nn.functional.log_softmax",
    "torch.nn.functional.cross_entropy",
    "torch.nn.functional.scaled_dot_product_attention",
    "torch.nn.utils.clip_grad_norm_",
    "torch.optim.Adam",
    "torch.optim.AdamW",
}


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _resolve(name: str, aliases: dict[str, str]) -> str:
    first, separator, remainder = name.partition(".")
    resolved = aliases.get(first, first)
    return f"{resolved}.{remainder}" if separator else resolved


def test_no_plainly_forbidden_high_level_implementations() -> None:
    violations: list[str] = []
    source_root = Path.cwd() / "src"
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_name(node.func)
            if dotted is None:
                continue
            resolved = _resolve(dotted, aliases)
            if resolved in BANNED_CALLS:
                violations.append(f"{path}:{node.lineno}: {resolved}")
    assert not violations, "forbidden high-level calls found:\n" + "\n".join(violations)
