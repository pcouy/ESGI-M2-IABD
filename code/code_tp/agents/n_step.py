from collections import deque
import numpy as np
import torch


class NStepBufferMixin:
    """
    Mixin to provide n-step transitions for replay buffers.

    This mixin adds n-step returns to a replay buffer by maintaining a small buffer
    of recent transitions. When the buffer is full, it calculates n-step returns
    and stores them in the main buffer.

    Example:
    ```
    class NStepReplayBuffer(NStepBufferMixin, ReplayBuffer):
        pass
    ```
    """

    def __init__(self, *args, n_step=1, gamma=0.99, **kwargs):
        """
        Initialize the n-step buffer mixin.

        Args:
            n_step (int): Number of steps to look ahead for returns
            gamma (float): Discount factor for future rewards
            *args, **kwargs: Arguments to pass to the parent class
        """
        self.n_step = n_step
        self.gamma = gamma
        # Initialize the n-step buffer as a deque with a fixed size
        self.n_step_buffer = deque(maxlen=n_step)
        super().__init__(*args, **kwargs)

    def _get_n_step_transition(self):
        n_step_reward = 0
        done = False
        next_state = self.n_step_buffer[-1][2]
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        prev_action = self.n_step_buffer[0][6]
        infos = self.n_step_buffer[0][5]
        for i, (_, _, _, r, terminal, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma**i) * r
            if terminal:
                done = True
                break
        return state, action, next_state, n_step_reward, done, infos, prev_action

    def store(self, state, action, next_state, reward, done, infos, prev_action=None):
        """
        Store a transition in the n-step buffer and update the main buffer if necessary.

        This method adds the current transition to the n-step buffer. When the buffer
        is full, it calculates the n-step return and stores the n-step transition
        in the main buffer.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting next state
            reward: Immediate reward
            done: Whether the episode has ended
            infos: Additional information
            prev_action: Previous action (if using previous actions)

        Returns:
            Result from the parent store method
        """
        # Add the current transition to the n-step buffer
        self.n_step_buffer.append(
            (state, action, next_state, reward, done, infos, prev_action)
        )

        # If the n-step buffer isn't full yet, return without storing in the main buffer
        if len(self.n_step_buffer) < self.n_step and not done:
            return None

        store_one_more = True
        while store_one_more and len(self.n_step_buffer) > 0:
            transition = self._get_n_step_transition()
            store_one_more = transition[4]
            self.n_step_buffer.popleft()
            # Store the n-step transition in the main buffer
            result = super().store(*transition)

        # If the episode has ended, clear the n-step buffer
        if done:
            self.n_step_buffer.clear()

        return result
