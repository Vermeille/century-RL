"""Sanity checks for the Transformer module using tiny MLM tasks."""

from boardrl.experiments.transformer import (
    ToyMLM,
    transformer_learns_alphabet_mlm,
    transformer_needs_context_mlm,
    transformer_positional,
)


def test_transformer_learns_alphabet_mlm_lpe():
    transformer_learns_alphabet_mlm(ToyMLM(dim=64))


def test_transformer_learns_alphabet_mlm_rotary():
    transformer_learns_alphabet_mlm(ToyMLM(dim=64, rotary=True))


def test_transformer_needs_context_mlm_lpe():
    transformer_needs_context_mlm(ToyMLM(dim=64))


def test_transformer_needs_context_mlm_rotary():
    transformer_needs_context_mlm(ToyMLM(dim=64, rotary=True))


def test_transformer_positional_lpe():
    transformer_positional(ToyMLM())


def test_transformer_positional_rotary():
    transformer_positional(ToyMLM(rotary=True))
