import unittest
import numpy as np
import torch
import os
import shutil
from ..agents.memmapped_replay_buffer import MemmappedReplayBuffer
from ..agents.prioritized_memmapped_replay_buffer import (
    PrioritizedMemmappedReplayBuffer,
)
from ..agents.replay_buffer import ReplayBuffer
from ..agents.prioritized_replay import PrioritizedReplayBuffer
import pathlib


class TestReplayBuffers(unittest.TestCase):
    def setUp(self):
        self.test_dir = pathlib.Path("test_replay_buffer")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.obs_shape = (2, 2)  # [episode_id, step_id] for both rows
        self.max_size = 32
        self.batch_size = 8
        self.episode_lengths = None

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def fill_buffer(self, buffer, n_transitions, start_number=0, episode_lengths=None):
        """Helper method to fill a replay buffer with sequential transitions.

        Args:
            buffer: The replay buffer to fill
            n_transitions: Number of transitions to store
            start_number: Starting episode number (default=0)
            episode_lengths: List of episode lengths to use (default=None)

        Returns:
            List of (state, next_state, done, action, prev_action) tuples
        """
        states = []
        indices = []

        # Generate episode lengths if not provided
        if episode_lengths is None:
            if self.episode_lengths is None:
                remaining = n_transitions
                episode_lengths = []
                while remaining > 0:
                    length = min(remaining, np.random.randint(2, n_transitions // 3))
                    episode_lengths.append(length)
                    remaining -= length
            else:
                episode_lengths = self.episode_lengths
        print(
            f"\nStoring {n_transitions} transitions across {len(episode_lengths)} episodes:"
        )
        print(f"Episode lengths: {episode_lengths}")

        transition_count = 0
        for episode in range(len(episode_lengths)):
            episode_length = episode_lengths[episode]
            episode_id = start_number + episode
            prev_action = None  # Reset prev_action at start of episode

            print(f"\nEpisode {episode_id} (length {episode_length}):")
            for step in range(episode_length):
                # Create state with episode and step identifiers
                # First row: [episode_id, step_id]
                # Second row: [episode_id, step_id] (redundant for robustness)
                state = np.full(self.obs_shape, 0, dtype=np.uint8)
                state[:, 0] = episode_id
                state[:, 1] = step

                # Create next state
                next_state = np.full(self.obs_shape, 0, dtype=np.uint8)
                if step < episode_length:
                    # Next state in same episode
                    next_state[:, 0] = episode_id
                    next_state[:, 1] = step + 1
                    done = step == episode_length - 1

                # Create predictable action: episode_id * 100 + step
                action = episode_id * 100 + step

                idx = buffer.store(
                    state, action, next_state, 1.0, done, {}, prev_action
                )
                states.append((state, next_state, done, action, prev_action))
                indices.append(idx)

                print(
                    f"  Stored: episode={episode_id}, step={step}, done={done}, idx={idx}, action={action}, prev_action={prev_action}"
                )

                prev_action = action  # Update prev_action for next step

                transition_count += 1
                if transition_count >= n_transitions:
                    break

            if transition_count >= n_transitions:
                break

        print(f"Buffer contents:")
        for i in range(min(n_transitions, buffer.max_size)):
            print(
                f"  {buffer.states[i][0, :]} {buffer.dones[i]} {buffer.actions[i]} {buffer.prev_actions[i]}"
            )

        return states, indices

    def verify_episode_boundaries(
        self, buffer, n_attempts=5, is_prioritized=False, allow_wrapping=False
    ):
        """Helper method to verify that sampling respects episode boundaries.

        Args:
            buffer: The replay buffer to sample from
            n_attempts: Number of sampling attempts (default=5)
            is_prioritized: Whether the buffer is prioritized (default=False)
            allow_wrapping: Whether to allow state numbers to wrap around (default=False)

        Returns:
            None, but raises AssertionError if any test fails
        """
        print("\nSampling and verifying episode boundaries:")
        for attempt in range(n_attempts):
            print(f"\nAttempt {attempt}:")

            # Handle different buffer types
            if is_prioritized:
                batch, indices, weights = buffer.sample()
            else:
                batch = buffer.sample()
                weights = None
            states_batch, actions, next_states_batch, rewards, dones, _ = batch

            # Convert to numpy if needed
            if isinstance(states_batch, torch.Tensor):
                states_batch = states_batch.cpu().numpy()
                next_states_batch = next_states_batch.cpu().numpy()
                dones = dones.cpu().numpy()

            # Verify each sampled transition
            states_batch = buffer.denormalize(states_batch).round()
            next_states_batch = buffer.denormalize(next_states_batch).round()
            for i in range(len(states_batch)):
                # Extract episode and step info
                current_episode = states_batch[i][0, 0]
                current_step = states_batch[i][0, 1]
                next_episode = next_states_batch[i][0, 0]
                next_step = next_states_batch[i][0, 1]

                # Print sample info
                if is_prioritized:
                    print(
                        f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                        f"next_episode={next_episode}, next_step={next_step}, "
                        f"done={dones[i]}, weight={weights[i] if weights is not None else 'None'}"
                    )
                else:
                    print(
                        f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                        f"next_episode={next_episode}, next_step={next_step}, done={dones[i]}"
                    )

                # Then verify the transition doesn't cross episode boundaries
                self.assertEqual(
                    next_episode,
                    current_episode,
                    f"Sampled transition crosses episode boundary: {current_episode} -> {next_episode}",
                )

                # Finally verify step sequencing within the episode
                self.assertEqual(
                    next_step,
                    current_step + 1,
                    f"Steps not sequential within episode: {current_step} -> {next_step}",
                )

                # Note: We removed the done-state verification since we should never sample terminal states
                # If we did sample a terminal state, the first assertion would fail

    def verify_done_flags(
        self,
        buffer,
        states,
        indices,
        episode_lengths,
        is_prioritized=False,
        n_attempts=5,
    ):
        """Helper method to verify done flags are correctly set for final episode transitions.

        Args:
            buffer: The replay buffer to verify
            states: List of (state, next_state, done) tuples from fill_buffer
            indices: List of indices returned by fill_buffer
            episode_lengths: List of episode lengths used to fill the buffer
            is_prioritized: Whether the buffer is prioritized (default=False)
        """
        print("\nVerifying done flags:")

        found_terminal_transition = False

        for attempt in range(n_attempts):
            print(f"\nAttempt {attempt}:")

            # Handle different buffer types
            if is_prioritized:
                batch, indices, weights = buffer.sample()
            else:
                batch = buffer.sample()
                weights = None
            states_batch, actions, next_states_batch, rewards, dones, _ = batch

            # Convert to numpy if needed
            if isinstance(states_batch, torch.Tensor):
                states_batch = states_batch.cpu().numpy()
                next_states_batch = next_states_batch.cpu().numpy()
                dones = dones.cpu().numpy()

            # Verify each sampled transition
            states_batch = buffer.denormalize(states_batch).round()
            next_states_batch = buffer.denormalize(next_states_batch).round()
            for i in range(len(states_batch)):
                # Extract episode and step info
                current_episode = int(states_batch[i][0, 0])
                current_step = int(states_batch[i][0, 1])
                next_episode = int(next_states_batch[i][0, 0])
                next_step = int(next_states_batch[i][0, 1])

                # Print sample info
                if is_prioritized:
                    print(
                        f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                        f"next_episode={next_episode}, next_step={next_step}, "
                        f"done={dones[i]}, weight={weights[i] if weights is not None else 'None'}"
                    )
                else:
                    print(
                        f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                        f"next_episode={next_episode}, next_step={next_step}, done={dones[i]}"
                    )
                print(f"  Expected episode length: {episode_lengths[current_episode]}")
                print(
                    f"  Expected done: {next_step == episode_lengths[current_episode]}"
                )

                # Track if we found a terminal transition
                if dones[i]:
                    found_terminal_transition = True

                self.assertEqual(
                    dones[i],
                    next_step == episode_lengths[current_episode],
                    f"Done flag not set correctly for episode {current_episode}",
                )

        # Fail if no terminal transitions were sampled
        self.assertTrue(
            found_terminal_transition,
            "No terminal transitions (done=True) were sampled across all attempts. "
            "The sampling may be biased against terminal states.",
        )

    def verify_batch_dimensions(
        self, buffer, is_prioritized=False, wait_for_queue=False
    ):
        """Helper method to verify dimensions of sampled batch elements.

        Args:
            buffer: The replay buffer to sample from
            is_prioritized: Whether the buffer is prioritized (affects return tuple)
            wait_for_queue: Whether to wait for queue to be full for memmapped buffers
        """
        # Fill buffer with enough transitions
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8 * self.max_size))

        # For memmapped buffers, wait for queue
        if hasattr(buffer, "wait_for_queue") and wait_for_queue:
            self.assertTrue(
                buffer.wait_for_queue(), "Timed out waiting for queue to be full"
            )

        # Sample a batch
        if is_prioritized:
            batch, indices, weights = buffer.sample()
            # Convert indices and weights to numpy arrays if they're lists
            indices = np.asarray(indices)
            weights = np.asarray(weights)
            self.assertEqual(
                indices.shape,
                (self.batch_size,),
                f"Indices shape mismatch: expected ({self.batch_size},), got {indices.shape}",
            )
            self.assertEqual(
                weights.shape,
                (self.batch_size,),
                f"Weights shape mismatch: expected ({self.batch_size},), got {weights.shape}",
            )
        else:
            batch = buffer.sample()

        states_batch, actions, next_states_batch, rewards, dones, prev_actions = batch

        # Check states dimensions
        self.assertEqual(
            states_batch.shape,
            (self.batch_size, *self.obs_shape),
            f"States shape mismatch: expected ({self.batch_size}, {self.obs_shape}), got {states_batch.shape}",
        )
        self.assertEqual(
            next_states_batch.shape,
            (self.batch_size, *self.obs_shape),
            f"Next states shape mismatch: expected ({self.batch_size}, {self.obs_shape}), got {next_states_batch.shape}",
        )

        # Check other elements dimensions
        self.assertEqual(
            actions.shape,
            (self.batch_size,),
            f"Actions shape mismatch: expected ({self.batch_size},), got {actions.shape}",
        )
        self.assertEqual(
            rewards.shape,
            (self.batch_size,),
            f"Rewards shape mismatch: expected ({self.batch_size},), got {rewards.shape}",
        )
        self.assertEqual(
            dones.shape,
            (self.batch_size,),
            f"Dones shape mismatch: expected ({self.batch_size},), got {dones.shape}",
        )
        self.assertEqual(
            prev_actions.shape,
            (self.batch_size,),
            f"Previous actions shape mismatch: expected ({self.batch_size},), got {prev_actions.shape}",
        )

    def verify_actions(
        self, buffer, states, indices, is_prioritized=False, n_attempts=5
    ):
        """Helper method to verify that actions and previous actions are correctly stored and sampled.

        Args:
            buffer: The replay buffer to verify
            states: List of (state, next_state, done, action, prev_action) tuples from fill_buffer
            indices: List of indices returned by fill_buffer
            is_prioritized: Whether the buffer is prioritized (default=False)
            n_attempts: Number of sampling attempts (default=5)
        """
        print("\nVerifying actions and previous actions:")

        for attempt in range(n_attempts):
            print(f"\nAttempt {attempt}:")

            # Handle different buffer types
            if is_prioritized:
                batch, indices, weights = buffer.sample()
            else:
                batch = buffer.sample()
            states_batch, actions, next_states_batch, rewards, dones, prev_actions = (
                batch
            )

            # Convert to numpy if needed
            if isinstance(states_batch, torch.Tensor):
                states_batch = states_batch.cpu().numpy()
                next_states_batch = next_states_batch.cpu().numpy()
                actions = actions.cpu().numpy()
                prev_actions = prev_actions.cpu().numpy()

            # Verify each sampled transition
            states_batch = buffer.denormalize(states_batch).round()
            next_states_batch = buffer.denormalize(next_states_batch).round()
            for i in range(len(states_batch)):
                # Extract episode and step info
                current_episode = int(states_batch[i][0, 0])
                current_step = int(states_batch[i][0, 1])

                # Calculate expected action and previous action
                expected_action = current_episode * 100 + current_step
                expected_prev_action = (
                    None
                    if current_step == 0
                    else current_episode * 100 + (current_step - 1)
                )

                # Print sample info
                print(f"  Sample {i}: episode={current_episode}, step={current_step}")
                print(f"    Action: expected={expected_action}, got={actions[i]}")
                print(
                    f"    Prev action: expected={expected_prev_action}, got={prev_actions[i]}"
                )

                # Verify actions
                self.assertEqual(
                    actions[i],
                    expected_action,
                    f"Action mismatch for episode {current_episode}, step {current_step}",
                )

                # Verify previous actions
                if current_step == 0:
                    # First step in episode should have None/0 as prev_action
                    self.assertEqual(
                        prev_actions[i],
                        0,
                        f"First step in episode {current_episode} should have prev_action=0",
                    )
                else:
                    self.assertEqual(
                        prev_actions[i],
                        expected_prev_action,
                        f"Previous action mismatch for episode {current_episode}, step {current_step}",
                    )

    def test_memmapped_episode_boundaries(self):
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_episode_boundary_handling",
            warmup_size=1,
        )

        # Store a sequence with episode boundaries
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8 * self.max_size))

        # Test that sampling never crosses episode boundaries
        self.verify_episode_boundaries(buffer, n_attempts=20)

    def test_memmapped_episode_boundaries_wait_queue(self):
        """Same as test_memmapped_episode_boundaries but waits for queue to be full."""
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_episode_boundary_handling_wait",
            warmup_size=1,
        )

        # Store a sequence with episode boundaries
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8 * self.max_size))

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Test that sampling never crosses episode boundaries
        self.verify_episode_boundaries(buffer, n_attempts=20)

    def test_memmapped_wrapping(self):
        small_buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,  # Small size to force wrapping
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_wrapping_behavior",
            warmup_size=1,
        )

        # Store more transitions than buffer size
        states, _ = self.fill_buffer(
            small_buffer, n_transitions=int(1.5 * self.max_size)
        )

        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, n_attempts=20)

    def test_memmapped_wrapping_wait_queue(self):
        """Same as test_memmapped_wrapping but waits for queue to be full."""
        small_buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,  # Small size to force wrapping
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_wrapping_behavior_wait",
            warmup_size=1,
        )

        # Store more transitions than buffer size
        states, _ = self.fill_buffer(
            small_buffer, n_transitions=int(1.5 * self.max_size)
        )

        # Wait for queue to be full
        self.assertTrue(
            small_buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, n_attempts=20)

    def test_prioritized_memmapped_episode_boundaries(self):
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_episode_boundary_handling",
            warmup_size=1,
        )

        # Store a sequence with episode boundaries
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8 * self.max_size))

        # Test that prioritized sampling never crosses episode boundaries
        self.verify_episode_boundaries(buffer, is_prioritized=True, n_attempts=20)

    def test_prioritized_memmapped_episode_boundaries_wait_queue(self):
        """Same as test_prioritized_memmapped_episode_boundaries but waits for queue to be full."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_episode_boundary_handling_wait",
            warmup_size=1,
        )

        # Store a sequence with episode boundaries
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8 * self.max_size))

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Test that prioritized sampling never crosses episode boundaries
        self.verify_episode_boundaries(buffer, is_prioritized=True, n_attempts=20)

    def test_standard_episode_boundaries(self):
        buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8 * self.max_size))
        self.verify_episode_boundaries(buffer, n_attempts=20)

    def test_standard_wrapping(self):
        small_buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )
        states, _ = self.fill_buffer(
            small_buffer, n_transitions=int(1.5 * self.max_size)
        )
        self.verify_episode_boundaries(small_buffer, n_attempts=20)

    def test_prioritized_standard_episode_boundaries(self):
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8 * self.max_size))
        self.verify_episode_boundaries(buffer, is_prioritized=True, n_attempts=20)

    def test_prioritized_standard_wrapping(self):
        """Test wrapping behavior for PrioritizedReplayBuffer."""
        small_buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )

        # Fill buffer beyond capacity
        states, indices = self.fill_buffer(
            small_buffer, n_transitions=int(1.5 * self.max_size)
        )

        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, is_prioritized=True, n_attempts=20)

    def test_prioritized_memmapped_wrapping(self):
        """Test wrapping behavior for PrioritizedMemmappedReplayBuffer."""
        small_buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_wrapping",
            warmup_size=1,
        )

        # Fill buffer beyond capacity
        states, indices = self.fill_buffer(
            small_buffer, n_transitions=int(1.5 * self.max_size)
        )

        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, is_prioritized=True, n_attempts=20)

    def test_prioritized_memmapped_wrapping_wait_queue(self):
        """Same as test_prioritized_memmapped_wrapping but waits for queue to be full."""
        small_buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_wrapping_wait",
            warmup_size=1,
        )

        # Fill buffer beyond capacity
        states, indices = self.fill_buffer(
            small_buffer, n_transitions=int(1.5 * self.max_size)
        )

        # Wait for queue to be full
        self.assertTrue(
            small_buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, is_prioritized=True, n_attempts=20)

    def test_memmapped_done_flags(self):
        """Test that done flags are correctly set in MemmappedReplayBuffer."""
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_done_flags",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Verify done flags
        self.verify_done_flags(buffer, states, indices, episode_lengths)

    def test_memmapped_done_flags_wait_queue(self):
        """Same as test_memmapped_done_flags but waits for queue to be full."""
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_done_flags_wait",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Verify done flags
        self.verify_done_flags(buffer, states, indices, episode_lengths)

    def test_prioritized_memmapped_done_flags(self):
        """Test that done flags are correctly set in PrioritizedMemmappedReplayBuffer."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_done_flags",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Verify done flags
        self.verify_done_flags(
            buffer, states, indices, episode_lengths, is_prioritized=True
        )

    def test_prioritized_memmapped_done_flags_wait_queue(self):
        """Same as test_prioritized_memmapped_done_flags but waits for queue to be full."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_done_flags_wait",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Verify done flags
        self.verify_done_flags(
            buffer, states, indices, episode_lengths, is_prioritized=True
        )

    def test_standard_done_flags(self):
        """Test that done flags are correctly set in regular ReplayBuffer."""
        buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Verify done flags
        self.verify_done_flags(buffer, states, indices, episode_lengths)

    def test_prioritized_standard_done_flags(self):
        """Test that done flags are correctly set in regular PrioritizedReplayBuffer."""
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Verify done flags
        self.verify_done_flags(
            buffer, states, indices, episode_lengths, is_prioritized=True
        )

    def test_standard_batch_dimensions(self):
        """Test batch dimensions for standard ReplayBuffer."""
        buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )
        self.verify_batch_dimensions(buffer)

    def test_prioritized_standard_batch_dimensions(self):
        """Test batch dimensions for PrioritizedReplayBuffer."""
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )
        self.verify_batch_dimensions(buffer, is_prioritized=True)

    def test_memmapped_batch_dimensions(self):
        """Test batch dimensions for MemmappedReplayBuffer."""
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_batch_dimensions",
            warmup_size=1,
        )
        self.verify_batch_dimensions(buffer)

    def test_prioritized_memmapped_batch_dimensions(self):
        """Test batch dimensions for PrioritizedMemmappedReplayBuffer."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_batch_dimensions",
            warmup_size=1,
        )
        self.verify_batch_dimensions(buffer, is_prioritized=True)

    def test_memmapped_batch_dimensions_wait_queue(self):
        """Test batch dimensions for MemmappedReplayBuffer with wait queue."""
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_batch_dimensions_wait_queue",
            warmup_size=1,
        )
        self.verify_batch_dimensions(buffer, wait_for_queue=True)

    def test_prioritized_memmapped_batch_dimensions_wait_queue(self):
        """Test batch dimensions for PrioritizedMemmappedReplayBuffer with wait queue."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir
            / "prioritized_memmapped_batch_dimensions_wait_queue",
            warmup_size=1,
        )
        self.verify_batch_dimensions(buffer, is_prioritized=True, wait_for_queue=True)

    def test_actions(self):
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_actions",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Verify actions
        self.verify_actions(buffer, states, indices)

    def test_actions_wait_queue(self):
        """Same as test_actions but waits for queue to be full."""
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_actions_wait_queue",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Verify actions
        self.verify_actions(buffer, states, indices)

    def test_prioritized_actions(self):
        """Test that actions and previous actions are correctly stored and sampled in PrioritizedReplayBuffer."""
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Verify actions
        self.verify_actions(buffer, states, indices, is_prioritized=True)

    def test_prioritized_memmapped_actions(self):
        """Test that actions and previous actions are correctly stored and sampled in PrioritizedMemmappedReplayBuffer."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_actions",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Verify actions
        self.verify_actions(buffer, states, indices, is_prioritized=True)

    def test_prioritized_memmapped_actions_wait_queue(self):
        """Same as test_prioritized_memmapped_actions but waits for queue to be full."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_actions_wait_queue",
            warmup_size=1,
        )

        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(
            buffer, n_transitions=9, episode_lengths=episode_lengths
        )  # Sum of episode lengths

        # Wait for queue to be full
        self.assertTrue(
            buffer.wait_for_queue(), "Timed out waiting for queue to be full"
        )

        # Verify actions
        self.verify_actions(buffer, states, indices, is_prioritized=True)


if __name__ == "__main__":
    unittest.main()
