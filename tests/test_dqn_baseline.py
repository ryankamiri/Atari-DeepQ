"""
T1.1-T1.5 tests: schedule, replay, DQN-family target math, dueling nets, PER, runner overrides, plotting, integration runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.algos.dqn import DQNAgent, compute_bootstrap_target, linear_schedule
from src.nets import DuelingMLPQNetwork, MLPQNetwork, combine_dueling_streams
from src.replay import PrioritizedReplayBuffer, ReplayBuffer
from src.utils import set_global_seeds


def _load_plot_results_module():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / "plot_results.py"
    mod_name = "plot_results_t16"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "dd, duel, per, expected",
    [
        (True, True, True, "DDQN+Dueling+PER"),
        (True, True, False, "DDQN+Dueling"),
        (True, False, True, "DDQN+PER"),
        (True, False, False, None),
        (False, True, True, None),
    ],
)
def test_infer_ablation_variant_breakout_stack(dd: bool, duel: bool, per: bool, expected: str | None):
    pr = _load_plot_results_module()
    cfg = {
        "algorithm": {"double_dqn": dd},
        "model": {"dueling": duel},
        "replay": {"prioritized": per},
    }
    assert pr.infer_ablation_variant(cfg) == expected


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
    batch = buf.sample(32)
    assert batch.obs.shape == (32, 128) and batch.obs.dtype == np.float32
    assert batch.actions.shape == (32,) and batch.actions.dtype == np.int64
    assert batch.rewards.shape == (32,) and batch.rewards.dtype == np.float32
    assert batch.next_obs.shape == (32, 128) and batch.next_obs.dtype == np.float32
    assert batch.dones.shape == (32,) and batch.dones.dtype == np.float32
    assert batch.indices.shape == (32,) and batch.indices.dtype == np.int64
    assert batch.weights.shape == (32,) and batch.weights.dtype == np.float32


def test_uniform_replay_returns_unit_weights():
    buf = ReplayBuffer(50, obs_shape=(8,))
    for _ in range(40):
        buf.add(np.zeros(8, np.float32), 0, 0.0, np.zeros(8, np.float32), False)
    b = buf.sample(16, beta=None)
    assert np.allclose(b.weights, 1.0)


def test_uniform_replay_update_priorities_is_noop():
    buf = ReplayBuffer(20, obs_shape=(4,))
    for i in range(15):
        buf.add(np.ones(4, np.float32) * i, 0, 0.0, np.ones(4, np.float32), False)
    np.random.seed(0)
    counts_before = np.bincount(buf.sample(200).indices, minlength=20)
    buf.update_priorities(np.array([0, 1], dtype=np.int64), np.array([1e9, 1e9], dtype=np.float32))
    np.random.seed(0)
    counts_after = np.bincount(buf.sample(200).indices, minlength=20)
    assert np.array_equal(counts_before, counts_after)


def test_per_sample_shapes_finite_normalized_weights():
    buf = PrioritizedReplayBuffer(100, obs_shape=(16,), alpha=0.6)
    for i in range(60):
        buf.add(
            np.random.randn(16).astype(np.float32),
            i % 3,
            0.0,
            np.random.randn(16).astype(np.float32),
            False,
        )
    batch = buf.sample(32, beta=0.5)
    assert batch.obs.shape == (32, 16)
    assert batch.indices.shape == (32,)
    assert np.all(np.isfinite(batch.weights))
    assert np.all(batch.weights > 0) and np.all(batch.weights <= 1.0 + 1e-5)
    assert abs(float(np.max(batch.weights)) - 1.0) < 1e-4


def test_per_priority_update_biases_sampling():
    """After boosting one slot's raw priority, it is sampled far more often (seeded)."""
    np.random.seed(123)
    cap = 32
    buf = PrioritizedReplayBuffer(cap, obs_shape=(4,), alpha=0.6)
    for i in range(cap):
        buf.add(np.full(4, float(i), np.float32), 0, 0.0, np.zeros(4, np.float32), False)
    buf.update_priorities(np.array([7], dtype=np.int64), np.array([1000.0], dtype=np.float64))
    np.random.seed(42)
    n = 8000
    batch = buf.sample(n, beta=0.4)
    frac_7 = np.mean(batch.indices == 7)
    # uniform would be about 1/cap
    assert frac_7 > 3.0 / cap, f"expected strong bias toward index 7, got frac={frac_7}"


def test_per_importance_weights_use_global_normalization_not_batch_max():
    """
    If a high-priority item is sampled while lower-probability items exist elsewhere in replay,
    its normalized IS weight should be strictly below 1.0.
    """
    buf = PrioritizedReplayBuffer(16, obs_shape=(4,), alpha=0.6)
    for _ in range(16):
        z = np.zeros(4, dtype=np.float32)
        buf.add(z, 0, 0.0, z, False)
    buf.update_priorities(np.array([7], dtype=np.int64), np.array([1000.0], dtype=np.float64))

    np.random.seed(0)
    batch = buf.sample(1, beta=0.4)
    assert int(batch.indices[0]) == 7
    assert 0.0 < float(batch.weights[0]) < 1.0


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
    result = agent.update(obs, actions, rewards, next_obs, dones)
    m = result.metrics
    required = {"loss", "q_mean", "target_mean", "td_abs_mean"}
    assert set(m.keys()) == required
    for k, v in m.items():
        assert isinstance(v, float) and np.isfinite(v), f"{k}={v}"
    assert result.td_abs.shape == (bs,) and result.td_abs.dtype == np.float32
    assert np.all(np.isfinite(result.td_abs))
    changed = any(not torch.equal(before[n], p) for n, p in agent.q_net.named_parameters())
    assert changed, "optimizer should have updated weights"


@pytest.mark.parametrize("double_dqn", [False, True])
@pytest.mark.parametrize("dueling", [False, True])
def test_dqn_update_with_importance_weights_all_modes(double_dqn: bool, dueling: bool):
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
    w = np.random.uniform(0.2, 1.0, size=bs).astype(np.float32)
    w = w / float(np.max(w))
    result = agent.update(obs, actions, rewards, next_obs, dones, weights=w)
    assert result.td_abs.shape == (bs,)
    assert np.all(np.isfinite(result.td_abs))
    for k, v in result.metrics.items():
        assert np.isfinite(v), f"{k}={v}"


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


@pytest.mark.integration
def test_ddqn_dueling_per_smoke_run_produces_artifacts(tmp_path):
    """T1.4: PER + Double DQN + dueling on single trainer path."""
    repo = Path(__file__).resolve().parent.parent
    yaml_text = f"""
algorithm:
  double_dqn: true
replay:
  prioritized: true
  alpha: 0.6
  beta_start: 0.4
  beta_end: 1.0
  beta_anneal_steps: 4000
  priority_eps: 1.0e-6
experiment:
  name: t1_4_per_integration
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
    cfg_path = tmp_path / "mini_per.yaml"
    cfg_path.write_text(yaml_text)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_experiment.py"), "--config", str(cfg_path)],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    runs = list(tmp_path.glob("runs/*/"))
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "metrics.csv").exists()
    text = (run_dir / "metrics.csv").read_text()
    assert "beta" in text.splitlines()[0]
    assert "is_weight_mean" in text.splitlines()[0]
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    eval_files = list((run_dir / "eval").glob("*.json"))
    assert len(eval_files) >= 1


@pytest.mark.integration
def test_run_experiment_seed_override_updates_snapshot_and_run_dir(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    yaml_text = f"""
algorithm:
  double_dqn: false
experiment:
  name: seed_override_check
  seed: 42
  device: cpu
  output_root: {tmp_path.as_posix()}
env:
  id: ALE/Pong-ram-v5
model:
  hidden_dims: [32, 32]
  dueling: false
train:
  total_steps: 800
  batch_size: 32
  replay_capacity: 2000
  learning_starts: 200
  train_freq: 4
  gamma: 0.99
  lr: 1e-3
  target_update_interval: 100
  checkpoint_interval: 800
  grad_clip_norm: 10.0
exploration:
  epsilon_start: 1.0
  epsilon_end: 0.1
  decay_steps: 600
eval:
  eval_interval: 400
  n_episodes: 1
  epsilon_eval: 0.05
log:
  csv_flush_interval: 1
replay:
  prioritized: false
  alpha: 0.6
  beta_start: 0.4
  beta_end: 1.0
  beta_anneal_steps: 200000
  priority_eps: 1.0e-6
"""
    cfg_path = tmp_path / "seed_base.yaml"
    cfg_path.write_text(yaml_text)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_experiment.py"),
            "--config",
            str(cfg_path),
            "--seed",
            "44",
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    runs = list(tmp_path.glob("runs/*seed44/"))
    assert len(runs) == 1
    run_dir = runs[0]
    snapshot = (run_dir / "config.yaml").read_text()
    assert "seed: 44" in snapshot


def test_plot_results_synthetic_aggregation_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    repo = Path(__file__).resolve().parent.parent
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "figs"
    runs_root.mkdir(parents=True, exist_ok=True)

    def _make_run(name: str, env_id: str, variant: str, seed: int, complete: bool):
        run_dir = runs_root / name
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "experiment": {"name": name, "seed": seed, "device": "cpu", "output_root": "artifacts"},
            "env": {"id": env_id},
            "train": {"total_steps": 2_000_000},
            "eval": {"eval_interval": 50_000, "n_episodes": 20, "epsilon_eval": 0.05},
            "algorithm": {"double_dqn": variant != "DQN"},
            "model": {"dueling": variant != "DQN"},
            "replay": {"prioritized": variant != "DQN"},
        }
        (run_dir / "config.yaml").write_text(json.dumps(cfg))
        steps = [50_000, 100_000, 2_000_000] if complete else [50_000, 100_000]
        for step in steps:
            payload = {
                "env_id": env_id,
                "n_episodes": 20,
                "mean_return": float(seed + step / 100_000.0),
                "std_return": 1.0,
            }
            (eval_dir / f"eval_step_{step}.json").write_text(json.dumps(payload))

    # complete runs (include two seeds for Pong DQN, one for other combinations)
    _make_run("pong_dqn_seed42", "ALE/Pong-ram-v5", "DQN", 42, True)
    _make_run("pong_dqn_seed43", "ALE/Pong-ram-v5", "DQN", 43, True)
    _make_run("pong_per_seed42", "ALE/Pong-ram-v5", "PER", 42, True)
    _make_run("breakout_dqn_seed42", "ALE/Breakout-ram-v5", "DQN", 42, True)
    _make_run("breakout_per_seed42", "ALE/Breakout-ram-v5", "PER", 42, True)
    _make_run("boxing_dqn_seed42", "ALE/Boxing-ram-v5", "DQN", 42, True)
    _make_run("boxing_per_seed42", "ALE/Boxing-ram-v5", "PER", 42, True)
    # incomplete run should be skipped
    _make_run("pong_dqn_incomplete", "ALE/Pong-ram-v5", "DQN", 44, False)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "plot_results.py"),
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert (out_dir / "learning_curve_pong.png").exists()
    assert (out_dir / "learning_curve_breakout.png").exists()
    assert (out_dir / "learning_curve_boxing.png").exists()
    assert (out_dir / "online_final_eval_table.csv").exists()
    assert (out_dir / "online_final_eval_table.md").exists()
    summary_path = out_dir / "online_matrix_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    included = [r["run_dir"] for r in summary["runs_included"]]
    assert not any("incomplete" in p for p in included)


def test_plot_results_breakout_ablation_synthetic_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    repo = Path(__file__).resolve().parent.parent
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "abl"
    runs_root.mkdir(parents=True, exist_ok=True)

    def _write_breakout_run(name: str, dd: bool, duel: bool, per: bool, seed: int, complete: bool):
        run_dir = runs_root / name
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "experiment": {"name": name, "seed": seed, "device": "cpu", "output_root": "artifacts"},
            "env": {"id": "ALE/Breakout-ram-v5"},
            "train": {"total_steps": 2_000_000},
            "eval": {"eval_interval": 50_000, "n_episodes": 20, "epsilon_eval": 0.05},
            "algorithm": {"double_dqn": dd},
            "model": {"dueling": duel},
            "replay": {"prioritized": per},
        }
        (run_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        steps = [50_000, 2_000_000] if complete else [50_000]
        for step in steps:
            mean_return = float(10 * seed + (0.5 if per else 0) + (0.25 if duel else 0) + step / 1_000_000)
            payload = {"mean_return": mean_return, "n_episodes": 20}
            (eval_dir / f"eval_step_{step}.json").write_text(json.dumps(payload))

    for seed in (42, 43, 44):
        _write_breakout_run(f"bo_main_{seed}", True, True, True, seed, True)
        _write_breakout_run(f"bo_duel_{seed}", True, True, False, seed, True)
        _write_breakout_run(f"bo_per_{seed}", True, False, True, seed, True)
    _write_breakout_run("bo_incomplete", True, True, True, 99, False)
    _write_breakout_run("pong_main_wrong_env", True, True, True, 42, True)
    (runs_root / "pong_main_wrong_env" / "config.yaml").write_text(
        yaml.dump(
            {
                "experiment": {"name": "x", "seed": 42, "device": "cpu", "output_root": "artifacts"},
                "env": {"id": "ALE/Pong-ram-v5"},
                "train": {"total_steps": 2_000_000},
                "eval": {"eval_interval": 50_000, "n_episodes": 20, "epsilon_eval": 0.05},
                "algorithm": {"double_dqn": True},
                "model": {"dueling": True},
                "replay": {"prioritized": True},
            },
            default_flow_style=False,
            sort_keys=False,
        )
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "plot_results.py"),
            "--report",
            "breakout_ablation",
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert (out_dir / "breakout_ablation_curves.png").exists()
    assert (out_dir / "breakout_ablation_final_table.csv").exists()
    assert (out_dir / "breakout_ablation_final_table.md").exists()
    summary = json.loads((out_dir / "breakout_ablation_summary.json").read_text())
    assert summary["report"] == "breakout_ablation"
    assert summary["status"] == "complete"
    assert summary["group_counts"]["DDQN+Dueling+PER"] == 3
    assert summary["group_counts"]["DDQN+Dueling"] == 3
    assert summary["group_counts"]["DDQN+PER"] == 3
    assert len(summary["runs_included"]) == 9
    dirs = {Path(x["run_dir"]).name for x in summary["runs_included"]}
    assert "bo_incomplete" not in dirs
    assert "pong_main_wrong_env" not in dirs


def test_plot_results_breakout_ablation_no_matching_runs_writes_status_outputs(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "abl"
    runs_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "plot_results.py"),
            "--report",
            "breakout_ablation",
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "No completed full-protocol Breakout ablation runs were found" in r.stdout
    assert not (out_dir / "breakout_ablation_curves.png").exists()
    summary = json.loads((out_dir / "breakout_ablation_summary.json").read_text())
    assert summary["status"] == "no_completed_runs"
    md_text = (out_dir / "breakout_ablation_final_table.md").read_text()
    assert "No completed full-protocol Breakout ablation runs were found." in md_text


def test_plot_results_no_matching_runs_writes_status_outputs(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "figs"
    runs_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "plot_results.py"),
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "No completed full-protocol headline runs were found" in r.stdout
    assert not any(out_dir.glob("learning_curve_*.png"))
    summary = json.loads((out_dir / "online_matrix_summary.json").read_text())
    assert summary["status"] == "no_completed_runs"
    md_text = (out_dir / "online_final_eval_table.md").read_text()
    assert "No completed full-protocol headline runs were found." in md_text
