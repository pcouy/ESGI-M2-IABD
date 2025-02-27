import unittest
import numpy as np
import torch
import os
import shutil
import pathlib
from ..agents.prioritized_memmapped_replay_buffer import (
    PrioritizedMemmappedReplayBuffer,
)


class TestPrioritizedMemmappedReplay(unittest.TestCase):
    def setUp(self):
        self.test_dir = pathlib.Path("test_prioritized_memmapped_replay")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.obs_shape = (2, 2)  # [episode_id, step_id] for both rows
        self.max_size = 32
        self.batch_size = 8

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_priority_based_sampling_frequency(self):
        """Test that transitions with higher priorities are sampled more frequently."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            alpha=1.0,  # No priority exponent for easier testing
            beta=0.0,  # No importance sampling for this test
            storage_path=self.test_dir / "priority_sampling_test",
            warmup_size=1,
        )

        # Create transitions with increasing priorities
        # Repeat the pattern to fill the buffer
        base_priorities = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        n_repeats = self.max_size // len(base_priorities) + 1
        priorities = np.tile(base_priorities, n_repeats)[: self.max_size]

        # Store transitions and set their priorities
        for i in range(self.max_size):
            state = np.full(self.obs_shape, i, dtype=np.uint8)
            next_state = np.full(self.obs_shape, i + 1, dtype=np.uint8)
            idx = buffer.store(state, i, next_state, 1.0, False, {})
            buffer.update(idx + buffer.tree.capacity - 1, priorities[i])

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Sample many times to check frequency distribution
        n_samples = 1000
        state_counts = {i: 0 for i in range(self.max_size)}

        for _ in range(n_samples):
            batch, _, _ = buffer.sample()
            sampled_states = buffer.denormalize(batch[0]).round()
            for state in sampled_states:
                state_id = int(state[0, 0])  # First element identifies the transition
                state_counts[state_id] += 1

        # Group states by their priority level and calculate average sampling frequency
        priority_level_counts = {p: [] for p in base_priorities}
        for i in range(self.max_size):
            priority = priorities[i]
            if priority in base_priorities:  # Only check the base priority levels
                priority_level_counts[priority].append(state_counts[i])

        # Calculate average count for each priority level
        priority_level_avgs = {
            p: np.mean(counts) for p, counts in priority_level_counts.items()
        }

        # Verify that higher priority states are sampled more frequently on average
        priority_levels = sorted(base_priorities)
        for i in range(len(priority_levels) - 1):
            self.assertGreater(
                priority_level_avgs[priority_levels[i + 1]],
                priority_level_avgs[priority_levels[i]],
                f"Priority level {priority_levels[i + 1]} was sampled less frequently "
                f"({priority_level_avgs[priority_levels[i + 1]]} times on average) than "
                f"priority level {priority_levels[i]} ({priority_level_avgs[priority_levels[i]]} "
                f"times on average)",
            )

    def test_priority_updates(self):
        """Test that priorities are correctly updated and affect sampling weights."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            alpha=1.0,  # No priority exponent for easier testing
            beta=1.0,  # Full importance sampling for weight testing
            default_error=1.0,  # Start with uniform priorities
            storage_path=self.test_dir / "priority_updates_test",
            warmup_size=1,
        )

        # Create transitions with repeating priority pattern
        base_priorities = [1.0, 2.0, 4.0]
        n_repeats = self.max_size // len(base_priorities) + 1
        priorities = np.tile(base_priorities, n_repeats)[: self.max_size]

        # Store transitions up to max size
        for i in range(self.max_size):
            state = np.full(self.obs_shape, i, dtype=np.uint8)
            next_state = np.full(self.obs_shape, i + 1, dtype=np.uint8)
            buffer.store(state, i, next_state, 1.0, False, {})

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Update priorities with repeating pattern
        for i in range(self.max_size):
            idx = i + buffer.tree.capacity - 1
            buffer.update(idx, priorities[i])

        # Sample and verify weights reflect the priorities
        batch, indices, weights = buffer.sample()

        # With beta=1, weights should be inversely proportional to priorities
        unique_priorities = np.array(base_priorities)
        normalized_priorities = unique_priorities / sum(unique_priorities)
        expected_weights = 1.0 / (len(unique_priorities) * normalized_priorities)
        expected_weights = expected_weights / np.max(expected_weights)

        # Get the actual weights for the sampled transitions
        actual_weights = []
        for idx in indices:
            idx = idx - buffer.tree.capacity + 1  # Convert to data index
            if idx >= 0 and idx < self.max_size:
                priority_idx = idx % len(base_priorities)
                if priorities[idx] == base_priorities[priority_idx]:
                    actual_weights.append(weights[len(actual_weights)])

        # Verify weights are correctly computed
        for w in actual_weights:
            self.assertTrue(
                any(abs(w - ew) < 1e-1 for ew in expected_weights),
                f"Weight {w} not found in expected weights {expected_weights}",
            )

    def test_zero_priority_handling(self):
        """Test that the buffer handles zero priorities gracefully."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            alpha=1.0,
            storage_path=self.test_dir / "zero_priority_test",
            warmup_size=1,
        )

        # Store transitions up to max size
        for i in range(self.max_size):
            state = np.full(self.obs_shape, i, dtype=np.uint8)
            next_state = np.full(self.obs_shape, i + 1, dtype=np.uint8)
            idx = buffer.store(state, i, next_state, 1.0, False, {})
            # Set priority to 0
            buffer.update(idx + buffer.tree.capacity - 1, 0)

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

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
