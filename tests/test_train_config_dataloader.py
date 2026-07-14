"""Tests for DataLoader defaults and kwargs helper (WP-BC sub-task B)."""

import torch

import train
from irodori_tts.config import TrainConfig


def test_trainconfig_dataloader_defaults():
    cfg = TrainConfig()
    assert cfg.num_workers == 8
    assert cfg.dataloader_persistent_workers is True
    assert cfg.dataloader_prefetch_factor == 4


def test_dataloader_kwargs_num_workers_zero_has_no_worker_only_keys():
    cfg = TrainConfig(num_workers=0)
    kwargs = train.dataloader_kwargs(
        cfg, torch.device("cpu"), collator=None, is_main_process=False
    )
    assert kwargs["num_workers"] == 0
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_dataloader_kwargs_num_workers_eight_sets_persistent_and_prefetch():
    cfg = TrainConfig(num_workers=8)
    kwargs = train.dataloader_kwargs(
        cfg, torch.device("cpu"), collator=None, is_main_process=True
    )
    assert kwargs["num_workers"] == 8
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_dataloader_kwargs_pin_memory_follows_device():
    cfg = TrainConfig()
    cpu_kwargs = train.dataloader_kwargs(
        cfg, torch.device("cpu"), collator=None, is_main_process=False
    )
    cuda_kwargs = train.dataloader_kwargs(
        cfg, torch.device("cuda"), collator=None, is_main_process=False
    )
    assert cpu_kwargs["pin_memory"] is False
    assert cuda_kwargs["pin_memory"] is True
