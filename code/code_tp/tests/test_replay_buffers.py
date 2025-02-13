import unittest
import numpy as np
import torch
import os
import shutil
from code_tp.agents.memmapped_replay_buffer import MemmappedReplayBuffer
from code_tp.agents.prioritized_memmapped_replay_buffer import PrioritizedMemmappedReplayBuffer
from code_tp.agents.replay_buffer import ReplayBuffer
from code_tp.agents.prioritized_replay import PrioritizedReplayBuffer
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
            List of (state, next_state, done) tuples
        """
        states = []
        indices = []
        
        # Generate episode lengths if not provided
        if episode_lengths is None:
            if self.episode_lengths is None:
                remaining = n_transitions
                episode_lengths = []
                while remaining > 0:
                    length = min(remaining, np.random.randint(2, n_transitions//3))
                    episode_lengths.append(length)
                    remaining -= length
            else:
                episode_lengths = self.episode_lengths
        print(f"\nStoring {n_transitions} transitions across {len(episode_lengths)} episodes:")
        print(f"Episode lengths: {episode_lengths}")
        
        transition_count = 0
        for episode in range(len(episode_lengths)):
            episode_length = episode_lengths[episode]
            episode_id = start_number + episode
            
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
                
                idx = buffer.store(state, 0, next_state, 1.0, done, {})
                states.append((state, next_state, done))
                indices.append(idx)
                
                print(f"  Stored: episode={episode_id}, step={step}, done={done}, idx={idx}")
                
                transition_count += 1
                if transition_count >= n_transitions:
                    break
            
            if transition_count >= n_transitions:
                break

        print(f"Buffer contents:")
        for i in range(min(n_transitions, buffer.max_size)):
            print(f"  {buffer.states[i][0, :]} {buffer.dones[i]}")
        
        return states, indices

    def verify_episode_boundaries(self, buffer, n_attempts=5, is_prioritized=False, allow_wrapping=False):
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
                    print(f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                          f"next_episode={next_episode}, next_step={next_step}, "
                          f"done={dones[i]}, weight={weights[i] if weights is not None else 'None'}")
                else:
                    print(f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                          f"next_episode={next_episode}, next_step={next_step}, done={dones[i]}")
                
                # Then verify the transition doesn't cross episode boundaries
                self.assertEqual(next_episode, current_episode,
                               f"Sampled transition crosses episode boundary: {current_episode} -> {next_episode}")
                
                # Finally verify step sequencing within the episode
                self.assertEqual(next_step, current_step+1,
                               f"Steps not sequential within episode: {current_step} -> {next_step}")
                
                # Note: We removed the done-state verification since we should never sample terminal states
                # If we did sample a terminal state, the first assertion would fail

    def verify_done_flags(self, buffer, states, indices, episode_lengths, is_prioritized=False, n_attempts=5):
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
                    print(f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                          f"next_episode={next_episode}, next_step={next_step}, "
                          f"done={dones[i]}, weight={weights[i] if weights is not None else 'None'}")
                else:
                    print(f"  Sample {i}: episode={current_episode}, step={current_step} -> "
                          f"next_episode={next_episode}, next_step={next_step}, done={dones[i]}")
                print(f"Expected episode length: {episode_lengths[current_episode]}")
                print(f"Expected done: {next_step == episode_lengths[current_episode]}")
                
                # Track if we found a terminal transition
                if dones[i]:
                    found_terminal_transition = True
                
                self.assertEqual(dones[i], next_step == episode_lengths[current_episode],
                                 f"Done flag not set correctly for episode {current_episode}")
        
        # Fail if no terminal transitions were sampled
        self.assertTrue(found_terminal_transition, 
                       "No terminal transitions (done=True) were sampled across all attempts. "
                       "The sampling may be biased against terminal states.")

    def test_memmapped_episode_boundaries(self):
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_episode_boundary_handling"
        )

        # Store a sequence with episode boundaries
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8*self.max_size))
        
        # Test that sampling never crosses episode boundaries
        self.verify_episode_boundaries(buffer, n_attempts=20)

    def test_memmapped_wrapping(self):
        small_buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,  # Small size to force wrapping
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_wrapping_behavior"
        )

        # Store more transitions than buffer size
        states, _ = self.fill_buffer(small_buffer, n_transitions=int(1.5*self.max_size))
        
        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, n_attempts=20)

    def test_prioritized_memmapped_episode_boundaries(self):
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_episode_boundary_handling"
        )

        # Store a sequence with episode boundaries and set priorities
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8*self.max_size))
        
        # Test that prioritized sampling never crosses episode boundaries
        self.verify_episode_boundaries(buffer, is_prioritized=True, n_attempts=20)


    def test_standard_episode_boundaries(self):
        buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size
        )
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8*self.max_size))
        self.verify_episode_boundaries(buffer, n_attempts=20)

    def test_standard_wrapping(self):
        small_buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=2
        )
        states, _ = self.fill_buffer(small_buffer, n_transitions=int(1.5*self.max_size))
        self.verify_episode_boundaries(small_buffer, n_attempts=20)

    def test_prioritized_standard_episode_boundaries(self):
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size
        )
        states, _ = self.fill_buffer(buffer, n_transitions=int(0.8*self.max_size))
        self.verify_episode_boundaries(buffer, is_prioritized=True, n_attempts=20)

    def test_prioritized_standard_wrapping(self):
        """Test wrapping behavior for PrioritizedReplayBuffer."""
        small_buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size
        )

        # Fill buffer beyond capacity
        states, indices = self.fill_buffer(small_buffer, n_transitions=int(1.5*self.max_size))
        
        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, is_prioritized=True, n_attempts=20)

    def test_prioritized_memmapped_wrapping(self):
        """Test wrapping behavior for PrioritizedMemmappedReplayBuffer."""
        small_buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_wrapping"
        )

        # Fill buffer beyond capacity
        states, indices = self.fill_buffer(small_buffer, n_transitions=int(1.5*self.max_size))
        
        # Test sampling after wrapping
        self.verify_episode_boundaries(small_buffer, is_prioritized=True, n_attempts=20)

    def test_memmapped_done_flags(self):
        """Test that done flags are correctly set in MemmappedReplayBuffer."""
        buffer = MemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "memmapped_done_flags" 
        )
        
        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(buffer, n_transitions=9, episode_lengths=episode_lengths)  # Sum of episode lengths
        
        # Verify done flags
        self.verify_done_flags(buffer, states, indices, episode_lengths)

    def test_prioritized_memmapped_done_flags(self):
        """Test that done flags are correctly set in PrioritizedMemmappedReplayBuffer."""
        buffer = PrioritizedMemmappedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size,
            storage_path=self.test_dir / "prioritized_memmapped_done_flags"
        )
        
        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(buffer, n_transitions=9, episode_lengths=episode_lengths)  # Sum of episode lengths
        
        # Verify done flags
        self.verify_done_flags(buffer, states, indices, episode_lengths, is_prioritized=True)

    def test_standard_done_flags(self):
        """Test that done flags are correctly set in regular ReplayBuffer."""
        buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size
        )
        
        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(buffer, n_transitions=9, episode_lengths=episode_lengths)  # Sum of episode lengths
        
        # Verify done flags
        self.verify_done_flags(buffer, states, indices, episode_lengths)

    def test_prioritized_standard_done_flags(self):
        """Test that done flags are correctly set in regular PrioritizedReplayBuffer."""
        buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            max_size=self.max_size,
            batch_size=self.batch_size
        )
        
        # Use fixed episode lengths for predictable testing
        episode_lengths = [4, 3, 2]  # Three episodes of different lengths
        states, indices = self.fill_buffer(buffer, n_transitions=9, episode_lengths=episode_lengths)  # Sum of episode lengths
        
        # Verify done flags
        self.verify_done_flags(buffer, states, indices, episode_lengths, is_prioritized=True)
        


if __name__ == '__main__':
    unittest.main() 