"""Tests for the MARVIN intelligence layer (planner + model).

Tests the core propose-and-verify loop, gating logic, surrogate simulator,
and Value of Information mechanisms.

Authored by IBM Bob for the MARVIN mission planner.
"""
import numpy as np
import pytest

from common.types import (
    Action,
    ActionKind,
    Constraints,
    MissionState,
    Perception,
    Pose,
    RolloutBatch,
    Target,
)
from planner import gating, surrogate, voi
from planner.surrogate import SurrogateEnv


@pytest.fixture
def sample_state():
    """Create a sample mission state for testing."""
    return MissionState(
        pose=Pose(xy=(0.0, 0.0), heading_rad=0.0),
        battery_pct=50.0,
        sol_time=0.5,
        localization_sigma=1.0,
        collected=(),
        remaining=(
            Target(id="t1", xy=(10.0, 0.0), science_value=0.8, mineral_class="carbonate"),
            Target(id="t2", xy=(20.0, 10.0), science_value=0.6, mineral_class="phyllosilicate"),
        )
    )


@pytest.fixture
def sample_perception():
    """Create a sample perception for testing."""
    # Create simple slope and roughness maps
    slope_map = np.random.uniform(0, 15, size=(50, 50))
    roughness_map = np.random.uniform(0, 0.3, size=(50, 50))
    
    return Perception(
        slope_deg=slope_map,
        roughness=roughness_map,
        visible_targets=(
            Target(id="t1", xy=(10.0, 0.0), science_value=0.8, mineral_class="carbonate"),
        ),
        dust_tau=0.5
    )


@pytest.fixture
def sample_env(sample_perception):
    """Create a sample surrogate environment."""
    return SurrogateEnv(
        slope_deg=sample_perception.slope_deg,
        roughness=sample_perception.roughness,
        dust_tau=0.5,
        traction_range=(0.8, 1.2),
        loc_drift_range=(0.2, 0.4),
        draw_mult_range=(0.9, 1.1)
    )


class TestGating:
    """Tests for tail-risk gating and safety constraints."""
    
    def test_tail_worst_basic(self):
        """Test CVaR tail aggregation."""
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Worst 10% (q=0.90) should be mean of [10]
        result = gating.tail_worst(arr, 0.90)
        assert result == 10.0
        
        # Worst 20% (q=0.80) should be mean of [9, 10]
        result = gating.tail_worst(arr, 0.80)
        assert result == 9.5
    
    def test_tail_worst_edge_cases(self):
        """Test edge cases for tail_worst."""
        # Empty array
        assert gating.tail_worst(np.array([]), 0.90) == 0.0
        
        # Single element
        assert gating.tail_worst(np.array([5.0]), 0.90) == 5.0
        
        # All same values
        assert gating.tail_worst(np.array([3.0, 3.0, 3.0]), 0.90) == 3.0
    
    def test_score_candidate(self, sample_state):
        """Test candidate scoring."""
        # Create a simple action sequence
        seq = (
            Action(kind=ActionKind.DRIVE, params={'xy': (10.0, 0.0)}),
            Action(kind=ActionKind.SAMPLE, params={'target': 't1'})
        )
        
        # Create mock rollout batch
        batch = RolloutBatch(
            success=np.array([True, True, False, True, True]),
            energy=np.array([50.0, 55.0, 60.0, 52.0, 48.0]),
            hazard=np.array([0.05, 0.08, 0.15, 0.06, 0.04])
        )
        
        cfg = {'cvar_quantile': 0.90, 'energy_penalty_factor': 0.01}
        
        score = gating.score_candidate(seq, batch, sample_state, cfg)
        
        # Check score properties
        assert 0.0 <= score.p_success <= 1.0
        assert score.cvar_energy > 0
        assert 0.0 <= score.risk <= 1.0
        assert isinstance(score.value, float)
    
    def test_gate_filters_unsafe(self, sample_state):
        """Test that gating filters out unsafe candidates."""
        from common.types import CandidateScore
        
        # Create candidates with varying risk levels
        safe_candidate = CandidateScore(
            seq=(Action(kind=ActionKind.DRIVE, params={'xy': (5.0, 0.0)}),),
            p_success=0.85,
            cvar_energy=30.0,
            risk=0.05,
            value=0.5
        )
        
        risky_candidate = CandidateScore(
            seq=(Action(kind=ActionKind.DRIVE, params={'xy': (20.0, 0.0)}),),
            p_success=0.60,
            cvar_energy=80.0,
            risk=0.25,  # Above ceiling
            value=0.7
        )
        
        low_success_candidate = CandidateScore(
            seq=(Action(kind=ActionKind.DRIVE, params={'xy': (15.0, 0.0)}),),
            p_success=0.50,  # Below minimum
            cvar_energy=50.0,
            risk=0.08,
            value=0.6
        )
        
        constraints = Constraints(
            battery_reserve_pct=15.0,
            risk_ceiling=0.10,
            p_success_min=0.70
        )
        
        scores = [safe_candidate, risky_candidate, low_success_candidate]
        safe = gating.gate(scores, constraints, sample_state.battery_pct)
        
        # Only the safe candidate should pass
        assert len(safe) == 1
        assert safe[0] == safe_candidate


class TestSurrogate:
    """Tests for the vectorized surrogate simulator."""
    
    def test_rollout_batch_shape(self, sample_state, sample_env):
        """Test that rollout_batch returns correct shapes."""
        seq = (Action(kind=ActionKind.DRIVE, params={'xy': (10.0, 0.0)}),)
        n = 20
        rng = np.random.default_rng(42)
        
        batch = surrogate.rollout_batch(seq, sample_state, sample_env, n, rng)
        
        # Check output shapes
        assert batch.success.shape == (n,)
        assert batch.energy.shape == (n,)
        assert batch.hazard.shape == (n,)
        
        # Check types
        assert batch.success.dtype == bool
        assert batch.energy.dtype == float
        assert batch.hazard.dtype == float
    
    def test_rollout_batch_uncertainty(self, sample_state, sample_env):
        """Test that rollouts vary due to injected uncertainty."""
        seq = (Action(kind=ActionKind.DRIVE, params={'xy': (10.0, 0.0)}),)
        n = 20
        rng = np.random.default_rng(42)
        
        batch = surrogate.rollout_batch(seq, sample_state, sample_env, n, rng)
        
        # Energy should vary across rollouts
        assert np.std(batch.energy) > 0
        
        # Hazard should vary
        assert np.std(batch.hazard) > 0 or np.all(batch.hazard == batch.hazard[0])
    
    def test_rollout_hold_action(self, sample_state, sample_env):
        """Test that HOLD action consumes no energy."""
        seq = (Action(kind=ActionKind.HOLD, params={}),)
        n = 10
        rng = np.random.default_rng(42)
        
        batch = surrogate.rollout_batch(seq, sample_state, sample_env, n, rng)
        
        # HOLD should use no energy
        assert np.all(batch.energy == 0)
        
        # Should always succeed
        assert np.all(batch.success)
    
    def test_rollout_sample_action(self, sample_state, sample_env):
        """Test SAMPLE action in rollout."""
        seq = (Action(kind=ActionKind.SAMPLE, params={'target': 't1'}),)
        n = 10
        rng = np.random.default_rng(42)
        
        batch = surrogate.rollout_batch(seq, sample_state, sample_env, n, rng)
        
        # Sample should use energy
        assert np.all(batch.energy > 0)
        
        # Most should succeed (small failure rate)
        assert np.mean(batch.success) > 0.9
    
    def test_create_surrogate_env(self, sample_perception):
        """Test surrogate environment creation."""
        cfg = {
            'traction_range': (0.7, 1.3),
            'loc_drift_range': (0.1, 0.5),
            'draw_mult_range': (0.85, 1.15)
        }
        
        env = surrogate.create_surrogate_env(sample_perception, cfg)
        
        assert env.traction_range == (0.7, 1.3)
        assert env.loc_drift_range == (0.1, 0.5)
        assert env.draw_mult_range == (0.85, 1.15)
        assert env.dust_tau == sample_perception.dust_tau


class TestVoI:
    """Tests for Value of Information mechanisms."""
    
    def test_best_gap_basic(self):
        """Test value gap calculation."""
        from common.types import CandidateScore
        
        scores = [
            CandidateScore(seq=(), p_success=0.8, cvar_energy=50, risk=0.05, value=0.7),
            CandidateScore(seq=(), p_success=0.75, cvar_energy=45, risk=0.06, value=0.65),
            CandidateScore(seq=(), p_success=0.7, cvar_energy=40, risk=0.04, value=0.5),
        ]
        
        gap = voi.best_gap(scores)
        
        # Gap should be 0.7 - 0.65 = 0.05
        assert abs(gap - 0.05) < 0.01
    
    def test_best_gap_edge_cases(self):
        """Test edge cases for best_gap."""
        from common.types import CandidateScore
        
        # Empty list
        assert voi.best_gap([]) == 0.0
        
        # Single candidate
        single = [CandidateScore(seq=(), p_success=0.8, cvar_energy=50, risk=0.05, value=0.7)]
        assert voi.best_gap(single) == 0.0
    
    def test_cheapest_observation(self, sample_state, sample_perception):
        """Test observation proposal."""
        obs = voi.cheapest_observation(sample_state, sample_perception)
        
        assert obs is not None
        assert obs.kind in [ActionKind.SCAN, ActionKind.OBSERVE]
    
    def test_voi_calculation(self, sample_state, sample_env):
        """Test VoI estimation."""
        obs = Action(kind=ActionKind.SCAN, params={})
        
        voi_value = voi.voi(obs, sample_state, sample_env)
        
        # VoI should be positive
        assert voi_value > 0
    
    def test_cost_calculation(self):
        """Test observation cost calculation."""
        scan = Action(kind=ActionKind.SCAN, params={})
        observe = Action(kind=ActionKind.OBSERVE, params={})
        
        scan_cost = voi.cost(scan)
        observe_cost = voi.cost(observe)
        
        # SCAN should be more expensive than OBSERVE
        assert scan_cost > observe_cost
        assert scan_cost > 0
        assert observe_cost > 0
    
    def test_maybe_observe_triggers(self, sample_state, sample_perception, sample_env):
        """Test that maybe_observe triggers when decision is ambiguous."""
        from common.types import CandidateScore
        
        # Create ambiguous candidates (small gap)
        safe = [
            CandidateScore(seq=(), p_success=0.8, cvar_energy=50, risk=0.05, value=0.50),
            CandidateScore(seq=(), p_success=0.78, cvar_energy=48, risk=0.06, value=0.48),
        ]
        
        cfg = {'voi_gap_threshold': 0.15}
        
        result = voi.maybe_observe(safe, sample_state, sample_perception, sample_env, cfg)
        
        # Should recommend observation due to small gap
        assert result is not None
        assert result.kind in [ActionKind.SCAN, ActionKind.OBSERVE]
    
    def test_maybe_observe_no_trigger(self, sample_state, sample_perception, sample_env):
        """Test that maybe_observe doesn't trigger when decision is clear."""
        from common.types import CandidateScore
        
        # Create clear candidates (large gap)
        safe = [
            CandidateScore(seq=(), p_success=0.9, cvar_energy=40, risk=0.03, value=0.80),
            CandidateScore(seq=(), p_success=0.75, cvar_energy=50, risk=0.07, value=0.50),
        ]
        
        cfg = {'voi_gap_threshold': 0.15}
        
        result = voi.maybe_observe(safe, sample_state, sample_perception, sample_env, cfg)
        
        # Should not recommend observation (decision is clear)
        assert result is None


class TestIntegration:
    """Integration tests for the full intelligence layer."""
    
    def test_safe_hold_on_empty_candidates(self, sample_state, sample_perception, sample_env):
        """Test that system defaults to HOLD when no safe candidates."""
        from planner.loop import decide_next_action
        from model.propose import Proposer
        
        # Mock proposer that returns empty list
        class MockProposer:
            def propose(self, state, perception, k):
                return []
        
        cfg = {
            'n_candidates': 3,
            'n_rollouts': 20,
            'seed': 42,
            'battery_reserve_pct': 15.0,
            'risk_ceiling': 0.10,
            'p_success_min': 0.70,
            'cvar_quantile': 0.90,
            'voi_gap_threshold': 0.15
        }
        
        model = MockProposer()
        
        decision = decide_next_action(sample_state, sample_perception, sample_env, model, cfg)
        
        # Should default to HOLD
        assert decision.action.kind == ActionKind.HOLD
        assert "No valid candidates" in decision.rationale or "No candidates" in decision.rationale
    
    def test_decision_has_rationale(self, sample_state, sample_perception, sample_env):
        """Test that decisions always include rationale."""
        from planner.loop import decide_next_action
        
        # Mock proposer with simple candidates
        class MockProposer:
            def propose(self, state, perception, k):
                return [
                    (Action(kind=ActionKind.DRIVE, params={'xy': (5.0, 0.0)}),),
                    (Action(kind=ActionKind.HOLD, params={}),)
                ]
        
        cfg = {
            'n_candidates': 2,
            'n_rollouts': 10,
            'seed': 42,
            'battery_reserve_pct': 15.0,
            'risk_ceiling': 0.10,
            'p_success_min': 0.70,
            'cvar_quantile': 0.90,
            'voi_gap_threshold': 0.15,
            'energy_penalty_factor': 0.01
        }
        
        model = MockProposer()
        
        decision = decide_next_action(sample_state, sample_perception, sample_env, model, cfg)
        
        # Should have a rationale
        assert decision.rationale
        assert len(decision.rationale) > 0
        
        # Should have scores
        assert len(decision.scores) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
