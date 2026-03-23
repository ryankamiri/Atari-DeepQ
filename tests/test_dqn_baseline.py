"""
T1.1/T1.2/T1.3 tests: schedule, replay, DQN-family target math, dueling nets, update metrics, integration runs.
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

from src.algos.dqn import DQNAgent, compute_bootstrap_target, linear_schedule
from src.nets import DuelingMLPQNetwork, MLPQNetwork, combine_dueling_streams
from src.replay import ReplayBuffer
from src.utils import set_global_seeds


def test_combine_dueling_streams_matches_v_plus_centered_advantage():
    value = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    advantage = torch.tensor([[0.0, 3.0, -3.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    q = combine_dueling_streams(value, advantage)
    expected = value + (advantage - advantage.mean(dim=1, keepdim=True))
    assert torch.allclose(q, expected)
    # row 0: mean A = 0 -> Q = 1 + [0,3,-3]
    assert torch.allclose(q[0], torch.tensor([1.0, 4.0, -2.0]))
    # row 1: mean A = 1 -> Q = 2 + [0,0,0]
    assert torch.allclose(q[1], torch.tensor([2.0, 2.0, 2.0]))


def test_dueling_mlp_forward_output_shape():
    net = DuelingMLPQNetwork(10, 5, [32, 32])
    x = torch.randn(7, 10)
    out = net(x)
    assert out.shape == (7, 5)


def test_dueling_mlp_empty_hidden_dims_heads_from_obs():
    net = DuelingMLPQNetwork(10, 4, [])
    x = torch.randn(3, 10)
    assert net(x).shape == (3, 4)


def test_dqn_agent_dueling_selects_network_class():
    device = torch.device("cpu")
    plain = DQNAgent(8, 4, [16], device=device, dueling=False)
    duel = DQNAgent(8, 4, [16], device=device, dueling=True)
    assert isinstance(plain.q_net, MLPQNetwork)
    assert isinstance(duel.q_net, DuelingMLPQNetwork)
    assert isinstance(plain.target_net, MLPQNetwork)
    assert isinstance(duel.target_net, DuelingMLPQNetwork)


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


def test_compute_bootstrap_target_vanilla_vs_double_dqn():
    """Vanilla uses max over target; Double uses online argmax then target gather."""
    # Batch size 2, 3 actions. Online argmax differs from target argmax.
    q_online = torch.tensor([[1.0, 10.0, 3.0], [5.0, 2.0, 8.0]], dtype=torch.float32)
    q_target = torch.tensor([[100.0, 20.0, 30.0], [1.0, 2.0, 300.0]], dtype=torch.float32)
    r = torch.zeros(2, dtype=torch.float32)
    d = torch.zeros(2, dtype=torch.float32)
    gamma = 0.99

    t_v = compute_bootstrap_target(q_target, r, d, gamma, double_dqn=False)
    # max target row0 = 100, row1 = 300
    assert torch.allclose(t_v, torch.tensor([99.0, 297.0]))

    t_dd = compute_bootstrap_target(q_target, r, d, gamma, double_dqn=True, q_online_next=q_online)
    # row0: online argmax action 1 -> target 20 -> 0.99*20 = 19.8
    # row1: online argmax action 2 -> target 300 -> 0.99*300 = 297
    assert torch.allclose(t_dd, torch.tensor([19.8, 297.0]))
    assert not torch.allclose(t_v, t_dd)


@pytest.mark.parametrize("double_dqn", [False, True])
@pytest.mark.parametrize("dueling", [False, True])
def test_dqn_update_returns_required_metrics_and_finite(double_dqn: bool, dueling: bool):
    device = torch.device("cpu")
    agent = DQNAgent(
        obs_dim=8,
        n_actions=4,
        hidden_dims=[32, 32],
        device=device,
        gamma=0.99,
        lr=1e-3,
        grad_clip_norm=10.0,
        double_dqn=double_dqn,
        dueling=dueling,
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
  dueling: false
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
algorithm:
  double_dqn: false
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


@pytest.mark.integration
def test_ddqn_smoke_run_produces_artifacts(tmp_path):
    """T1.2: end-to-end with algorithm.double_dqn: true."""
    repo = Path(__file__).resolve().parent.parent
    yaml_text = f"""
algorithm:
  double_dqn: true
experiment:
  name: t1_2_ddqn_integration
  seed: 0
  device: cpu
  output_root: {tmp_path.as_posix()}
env:
  id: ALE/Pong-ram-v5
model:
  hidden_dims: [32, 32]
  dueling: false
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
    cfg_path = tmp_path / "mini_ddqn.yaml"
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


@pytest.mark.integration
def test_ddqn_dueling_smoke_run_produces_artifacts(tmp_path):
    """T1.3: end-to-end with double_dqn and model.dueling true."""
    repo = Path(__file__).resolve().parent.parent
    yaml_text = f"""
algorithm:
  double_dqn: true
experiment:
  name: t1_3_ddqn_dueling_integration
  seed: 0
  device: cpu
  output_root: {tmp_path.as_posix()}
env:
  id: ALE/Pong-ram-v5
model:
  hidden_dims: [32, 32]
  dueling: true
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
    cfg_path = tmp_path / "mini_ddqn_dueling.yaml"
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
