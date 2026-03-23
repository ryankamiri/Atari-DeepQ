"""
T1.1 unit tests: epsilon schedule, replay buffer, DQN update metrics, target sync.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.algos.dqn import DQNAgent, linear_schedule
from src.replay import ReplayBuffer
from src.utils import set_global_seeds


def test_linear_schedule_monotonic_decay_to_end():
    start, end, decay = 1.0, 0.05, 1000
    prev = linear_schedule(0, start, end, decay)
    for step in range(1, decay + 500):
        cur = linear_schedule(step, start, end, decay)
        assert cur <= prev + 1e-9, f"not monotonic at step {step}"
        prev = cur
    assert abs(linear_schedule(0, start, end, decay) - start) < 1e-6
    assert abs(linear_schedule(decay, start, end, decay) - end) < 1e-6
    assert abs(linear_schedule(decay * 10, start, end, decay) - end) < 1e-6


def test_linear_schedule_zero_decay_returns_end():
    assert linear_schedule(0, 1.0, 0.05, 0) == 0.05


def test_replay_buffer_sample_shapes_and_dtypes():
    buf = ReplayBuffer(100, obs_shape=(128,))
    for i in range(50):
        s = np.random.rand(128).astype(np.float32)
        s2 = np.random.rand(128).astype(np.float32)
        buf.add(s, i % 6, float(i % 3 - 1), s2, i % 10 == 0)
    o, a, r, no, d = buf.sample(32)
    assert o.shape == (32, 128) and o.dtype == np.float32
    assert a.shape == (32,) and a.dtype == np.int64
    assert r.shape == (32,) and r.dtype == np.float32
    assert no.shape == (32, 128) and no.dtype == np.float32
    assert d.shape == (32,) and d.dtype == np.float32


def test_dqn_update_returns_required_metrics_and_finite():
    device = torch.device("cpu")
    agent = DQNAgent(
        obs_dim=8,
        n_actions=4,
        hidden_dims=[32, 32],
        device=device,
        gamma=0.99,
        lr=1e-3,
        grad_clip_norm=10.0,
    )
    bs = 16
    obs = np.random.rand(bs, 8).astype(np.float32)
    next_obs = np.random.rand(bs, 8).astype(np.float32)
    actions = np.random.randint(0, 4, size=bs, dtype=np.int64)
    rewards = np.random.randn(bs).astype(np.float32) * 0.1
    dones = np.zeros(bs, dtype=np.float32)
    before = {n: p.clone() for n, p in agent.q_net.named_parameters()}
    m = agent.update(obs, actions, rewards, next_obs, dones)
    required = {"loss", "q_mean", "target_mean", "td_abs_mean"}
    assert set(m.keys()) == required
    for k, v in m.items():
        assert isinstance(v, float) and np.isfinite(v), f"{k}={v}"
    changed = any(not torch.equal(before[n], p) for n, p in agent.q_net.named_parameters())
    assert changed, "optimizer should have updated weights"


def test_sync_target_copies_q_weights_exactly():
    device = torch.device("cpu")
    agent = DQNAgent(4, 2, [16], device=device, grad_clip_norm=10.0)
    with torch.no_grad():
        for p in agent.q_net.parameters():
            p.normal_(0, 0.5)
    agent.sync_target()
    for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(qp, tp)


def test_set_global_seeds_repeats_numpy_and_torch_sequences():
    set_global_seeds(123)
    np_seq_a = np.random.rand(4)
    torch_seq_a = torch.rand(4)

    set_global_seeds(123)
    np_seq_b = np.random.rand(4)
    torch_seq_b = torch.rand(4)

    assert np.allclose(np_seq_a, np_seq_b)
    assert torch.allclose(torch_seq_a, torch_seq_b)


@pytest.mark.integration
def test_smoke_run_produces_artifacts(tmp_path):
    """Tiny end-to-end run after T1.1 refactor: CSV, checkpoint, eval JSON."""
    repo = Path(__file__).resolve().parent.parent
    yaml_text = f"""
experiment:
  name: t1_1_integration
  seed: 0
  device: cpu
  output_root: {tmp_path.as_posix()}
env:
  id: ALE/Pong-ram-v5
model:
  hidden_dims: [32, 32]
train:
  total_steps: 2500
  batch_size: 32
  replay_capacity: 5000
  learning_starts: 500
  train_freq: 4
  gamma: 0.99
  lr: 1e-3
  target_update_interval: 200
  checkpoint_interval: 2500
  grad_clip_norm: 10.0
exploration:
  epsilon_start: 1.0
  epsilon_end: 0.1
  decay_steps: 2000
eval:
  eval_interval: 1000
  n_episodes: 1
  epsilon_eval: 0.05
log:
  csv_flush_interval: 1
"""
    cfg_path = tmp_path / "mini.yaml"
    cfg_path.write_text(yaml_text)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_experiment.py"), "--config", str(cfg_path)],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    runs = list(tmp_path.glob("runs/*/"))
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    eval_files = list((run_dir / "eval").glob("*.json"))
    assert len(eval_files) >= 1
    data = json.loads(eval_files[0].read_text())
    assert "mean_return" in data
