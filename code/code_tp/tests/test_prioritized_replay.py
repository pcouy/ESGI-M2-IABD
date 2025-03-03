import unittest
import numpy as np
import torch
from ..agents.prioritized_replay import PrioritizedReplayBuffer


class TestPrioritizedReplay(unittest.TestCase):
    def setUp(self):
        self.obs_shape = (2, 2)  # [episode_id, step_id] for both rows
        self.max_size = 32
        self.batch_size = 8

    def test_priority_based_sampling_frequency(self):
        """Test that transitions with higher priorities are sampled more frequently."""
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            alpha=1.0,  # No priority exponent for easier testing
            beta=0.0,  # No importance sampling for this test
            warmup_size=1,
        )

        # Create transitions with increasing priorities
        n_transitions = 5
        priorities = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

        # Store transitions and set their priorities
        for i in range(n_transitions):
            state = np.full(self.obs_shape, i, dtype=np.uint8)
            next_state = np.full(self.obs_shape, i + 1, dtype=np.uint8)
            idx = buffer.store(state, i, next_state, 1.0, False, {})
            buffer.update(idx + buffer.tree.capacity - 1, priorities[i])

        # Sample many times to check frequency distribution
        n_samples = 1000
        state_counts = {i: 0 for i in range(n_transitions)}

        for _ in range(n_samples):
            batch, _, _ = buffer.sample()
            sampled_states = buffer.denormalize(batch[0]).round()
            for state in sampled_states:
                state_id = int(state[0, 0])  # First element identifies the transition
                state_counts[state_id] += 1

        # Verify that higher priority states are sampled more frequently
        for i in range(n_transitions - 1):
            self.assertGreater(
                state_counts[i + 1],
                state_counts[i],
                f"State {i + 1} with priority {priorities[i + 1]} was sampled less frequently "
                f"({state_counts[i + 1]} times) than state {i} with priority {priorities[i]} "
                f"({state_counts[i]} times)",
            )

    def test_priority_updates(self):
        """Test that priorities are correctly updated and affect sampling weights."""
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            alpha=1.0,  # No priority exponent for easier testing
            beta=1.0,  # Full importance sampling for weight testing
            default_error=1.0,  # Start with uniform priorities
            warmup_size=1,
        )

        # Store several transitions
        n_transitions = 3
        for i in range(n_transitions):
            state = np.full(self.obs_shape, i, dtype=np.uint8)
            next_state = np.full(self.obs_shape, i + 1, dtype=np.uint8)
            buffer.store(state, i, next_state, 1.0, False, {})

        # Update priorities with known values
        priorities = [1.0, 2.0, 4.0]
        for i in range(n_transitions):
            idx = i + buffer.tree.capacity - 1
            buffer.update(idx, priorities[i])

        # Sample and verify weights reflect the priorities
        batch, indices, weights = buffer.sample()

        # With beta=1, weights should be inversely proportional to priorities
        normalized_priorities = np.array(priorities) / sum(priorities)
        expected_weights = 1.0 / (len(priorities) * normalized_priorities)
        expected_weights = expected_weights / np.max(expected_weights)

        # Get the actual weights for the sampled transitions
        actual_weights = []
        for idx in indices:
            idx = idx - buffer.tree.capacity + 1  # Convert to data index
            if idx >= 0 and idx < len(priorities):
                actual_weights.append(weights[len(actual_weights)])

        # Verify weights are correctly computed
        for w in actual_weights:
            self.assertTrue(
                any(abs(w - ew) < 1e-1 for ew in expected_weights),
                f"Weight {w} not found in expected weights {expected_weights}",
            )

    def test_zero_priority_handling(self):
        """Test that the buffer handles zero priorities gracefully."""
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            alpha=1.0,
            warmup_size=1,
        )

        # Store transitions
        n_transitions = 3
        for i in range(n_transitions):
            state = np.full(self.obs_shape, i, dtype=np.uint8)
            next_state = np.full(self.obs_shape, i + 1, dtype=np.uint8)
            idx = buffer.store(state, i, next_state, 1.0, False, {})
            # Set priority to 0
            buffer.update(idx + buffer.tree.capacity - 1, 0)

        # Should still be able to sample without errors
        batch, _, weights = buffer.sample()

        # Verify dimensions and values
        self.assertEqual(
            weights.shape, (self.batch_size,), "Importance weights shape mismatch"
        )
        self.assertTrue(
            torch.all(weights == 1.0),
            "All weights should be 1.0 when all priorities are zero",
        )


if __name__ == "__main__":
    unittest.main()
