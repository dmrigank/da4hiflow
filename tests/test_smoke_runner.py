"""Smoke tests for the runner."""

import numpy as np
from da4hiflow.core.runner import DummySystem, DummyAssimilator, Runner


def test_dummy_system():
    """Test dummy system."""
    system = DummySystem(state_size=5, seed=42)
    initial = system.get_initial_state()

    assert initial.shape == (5,)
    assert np.isfinite(initial).all()

    next_state = system.step(initial)
    assert next_state.shape == (5,)
    assert np.allclose(next_state, 1.1 * initial)


def test_dummy_assimilator():
    """Test dummy assimilator."""
    assim = DummyAssimilator(weight=0.5)
    state = np.ones(5)
    obs = np.zeros(5)

    result = assim.assimilate(state, obs)
    expected = 0.5 * state + 0.5 * obs

    assert np.allclose(result, expected)


def test_runner_smoke_test():
    """Smoke test: run the DA loop for 3 steps."""
    system = DummySystem(state_size=5, seed=42)
    assimilator = DummyAssimilator(weight=0.5)
    runner = Runner(system, assimilator, num_steps=3)

    trajectory = runner.run()

    # Check trajectory length: initial + 3 steps = 4 states
    assert len(trajectory) == 4, f"Expected 4 states, got {len(trajectory)}"

    # Check each state is the right shape
    for i, state in enumerate(trajectory):
        assert state.shape == (5,), f"State {i} has wrong shape: {state.shape}"
        assert np.isfinite(state).all(), f"State {i} has NaN/Inf values"

    # Check that trajectory is evolving (not all zeros)
    all_zeros = all(np.allclose(s, 0.0) for s in trajectory)
    assert not all_zeros, "Trajectory is all zeros"
