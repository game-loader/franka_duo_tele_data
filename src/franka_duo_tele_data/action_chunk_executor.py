#!/usr/bin/env python3
"""Action chunk executor with temporal consistency for policy evaluation.

When a policy outputs multiple future actions (action chunk/horizon), this
executor manages the buffer and decides when to request new inference.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ActionChunkExecutor:
    """Execute action chunks with configurable overlap and temporal blending.

    A typical workflow:
    1. Model inference returns [horizon, action_dim] chunk (e.g., 16 actions)
    2. Executor serves actions one by one at the control frequency (e.g., 30Hz)
    3. When reaching the overlap threshold, signal for new inference
    4. Optionally blend overlapping predictions for smoother transitions
    """

    def __init__(
        self,
        chunk_horizon: int = 16,
        overlap: int = 8,
        blend_overlap: bool = False,
    ):
        """Initialize action chunk executor.

        Args:
            chunk_horizon: Number of actions in each predicted chunk
            overlap: Request new chunk after consuming this many actions
            blend_overlap: If True, blend overlapping predictions from consecutive chunks
        """
        if chunk_horizon <= 0:
            raise ValueError("chunk_horizon must be positive")
        if not 0 < overlap < chunk_horizon:
            raise ValueError(f"overlap must be in (0, {chunk_horizon})")

        self.chunk_horizon = chunk_horizon
        self.overlap = overlap
        self.blend_overlap = blend_overlap
        self.action_buffer = deque(maxlen=chunk_horizon)
        self.next_chunk_buffer = deque(maxlen=chunk_horizon) if blend_overlap else None
        self.step_count = 0
        self._has_next_chunk = False
        self._signal_pending = False

    def add_chunk(self, actions: np.ndarray) -> None:
        """Add a new action chunk to the buffer.

        Args:
            actions: Action chunk with shape (chunk_horizon, action_dim)
        """
        if actions.ndim != 2:
            raise ValueError(f"Expected 2D action chunk, got shape {actions.shape}")
        if actions.shape[0] != self.chunk_horizon:
            raise ValueError(
                f"Expected chunk size {self.chunk_horizon}, got {actions.shape[0]}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Action chunk contains non-finite values")

        # If blending and we already have a current chunk, the new chunk becomes "next"
        if self.blend_overlap and len(self.action_buffer) > 0 and not self._has_next_chunk:
            self.next_chunk_buffer.clear()
            for action in actions:
                self.next_chunk_buffer.append(action.copy())
            self._has_next_chunk = True
        else:
            # First chunk or replace mode: clear and add
            self.action_buffer.clear()
            for action in actions:
                self.action_buffer.append(action.copy())
            self.step_count = 0
            self._has_next_chunk = False
            self._signal_pending = False

    def get_next_action(self) -> tuple[np.ndarray, bool]:
        """Get the next action to execute.

        Returns:
            action: The action to execute now (action_dim,)
            need_inference: True if a new chunk should be predicted
        """
        if len(self.action_buffer) == 0:
            raise RuntimeError(
                "Action buffer is empty. Call add_chunk() before get_next_action()."
            )

        current_index = self.step_count

        # Blend overlapping predictions if enabled and next chunk is available
        if (
            self.blend_overlap
            and self._has_next_chunk
            and current_index >= self.overlap
        ):
            # Temporal blending weight: linearly increase next chunk's influence
            blend_steps = self.chunk_horizon - self.overlap
            steps_into_overlap = current_index - self.overlap
            next_weight = min(1.0, steps_into_overlap / float(blend_steps))
            current_weight = 1.0 - next_weight

            current_action = self.action_buffer[current_index]
            next_action = self.next_chunk_buffer[current_index - self.overlap]
            action = current_weight * current_action + next_weight * next_action
        else:
            action = self.action_buffer[current_index]

        self.step_count += 1

        # Signal exactly once, when the overlap threshold is first crossed.
        need_inference = self.step_count == self.overlap and not self._has_next_chunk
        if self._signal_pending:
            # First action served from a switched-in chunk: its overlap
            # threshold was already consumed during blending, so ask now.
            need_inference = True
            self._signal_pending = False

        # If we've consumed the entire current chunk, switch to next if available
        if self.step_count >= self.chunk_horizon:
            if self._has_next_chunk and self.blend_overlap:
                self.action_buffer.clear()
                for buffered_action in self.next_chunk_buffer:
                    self.action_buffer.append(buffered_action)
                self.step_count = self.overlap  # Already consumed overlap actions
                self._has_next_chunk = False
                self._signal_pending = True
            else:
                # No next chunk, stay at the last action (policy should have provided new chunk)
                self.step_count = self.chunk_horizon - 1

        return np.ascontiguousarray(action, dtype=np.float32), need_inference

    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self.action_buffer.clear()
        if self.next_chunk_buffer is not None:
            self.next_chunk_buffer.clear()
        self.step_count = 0
        self._has_next_chunk = False
        self._signal_pending = False

    @property
    def is_empty(self) -> bool:
        """Check if the executor has no actions to serve."""
        return len(self.action_buffer) == 0

    @property
    def remaining_actions(self) -> int:
        """Number of actions remaining in current chunk."""
        if len(self.action_buffer) == 0:
            return 0
        return max(0, self.chunk_horizon - self.step_count)
