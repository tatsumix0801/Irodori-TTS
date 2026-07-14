"""CPU-only tests for train.sanitize_gradients_ (WP-BC sub-task C)."""

import torch

import train


def test_replaces_nonfinite_with_zero_and_keeps_finite():
    m = torch.nn.Linear(2, 2)
    m.weight.grad = torch.tensor(
        [[float("nan"), float("inf")], [float("-inf"), 7.5]]
    )
    m.bias.grad = torch.tensor([1.5, -2.25])

    train.sanitize_gradients_(m.parameters())

    assert m.weight.grad[0, 0].item() == 0.0
    assert m.weight.grad[0, 1].item() == 0.0
    assert m.weight.grad[1, 0].item() == 0.0
    assert m.weight.grad[1, 1].item() == 7.5
    assert torch.equal(m.bias.grad, torch.tensor([1.5, -2.25]))


def test_finite_grads_are_bit_identical():
    m = torch.nn.Linear(3, 3)
    original = torch.randn(3, 3)
    m.weight.grad = original.clone()

    train.sanitize_gradients_(m.parameters())

    assert torch.equal(m.weight.grad, original)


def test_sanitization_is_in_place():
    m = torch.nn.Linear(2, 2)
    g = torch.tensor([[float("nan"), 1.0], [2.0, float("inf")]])
    m.weight.grad = g

    train.sanitize_gradients_(m.parameters())

    assert m.weight.grad is g  # same tensor object, mutated in place


def test_none_grad_does_not_crash():
    m = torch.nn.Linear(2, 2)
    m.weight.grad = torch.tensor([[1.0, float("nan")], [0.0, 0.0]])
    # m.bias.grad intentionally left as None

    train.sanitize_gradients_(m.parameters())

    assert m.bias.grad is None
    assert m.weight.grad[0, 1].item() == 0.0
