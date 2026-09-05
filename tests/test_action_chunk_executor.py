#!/usr/bin/env python3
"""Unit tests for action chunk executor."""

from __future__ import annotations

import numpy as np
import pytest

from franka_duo_tele_data.action_chunk_executor import ActionChunkExecutor


def test_basic_chunk_execution():
    """Test basic action chunk execution without blending."""
    executor = ActionChunkExecutor(chunk_horizon=4, overlap=2, blend_overlap=False)

    # Create a simple chunk
    chunk = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    executor.add_chunk(chunk)

    # Execute first two actions (before overlap threshold)
    action1, need_inference1 = executor.get_next_action()
    np.testing.assert_array_equal(action1, [1, 2])
    assert not need_inference1

    action2, need_inference2 = executor.get_next_action()
    np.testing.assert_array_equal(action2, [3, 4])
    assert need_inference2  # Reached overlap threshold

    # Execute remaining actions
    action3, need_inference3 = executor.get_next_action()
    np.testing.assert_array_equal(action3, [5, 6])
    assert not need_inference3  # Already signaled

    action4, need_inference4 = executor.get_next_action()
    np.testing.assert_array_equal(action4, [7, 8])
    assert not need_inference4


def test_chunk_blending():
    """Test temporal blending of overlapping chunks."""
    executor = ActionChunkExecutor(chunk_horizon=4, overlap=2, blend_overlap=True)

    # First chunk
    chunk1 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    executor.add_chunk(chunk1)

    # Execute first two actions
    action1, need_inference1 = executor.get_next_action()
    np.testing.assert_array_equal(action1, [1, 1])
    assert not need_inference1

    action2, need_inference2 = executor.get_next_action()
    np.testing.assert_array_equal(action2, [2, 2])
    assert need_inference2

    # Add second chunk (should be buffered for blending)
    chunk2 = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    executor.add_chunk(chunk2)

    # Actions 3 and 4 should be blended between chunk1 and chunk2
    action3, need_inference3 = executor.get_next_action()
    # At step 2, blend between chunk1[2]=[3,3] and chunk2[0]=[10,10]
    # Blend weight increases from 0 to 1 over 2 steps
    # step_into_overlap=0, next_weight=0, current_weight=1
    np.testing.assert_array_equal(action3, [3, 3])

    action4, need_inference4 = executor.get_next_action()
    # step_into_overlap=1, next_weight=0.5, current_weight=0.5
    expected = 0.5 * np.array([4, 4]) + 0.5 * np.array([20, 20])
    np.testing.assert_allclose(action4, expected)

    # After consuming chunk1, should switch to chunk2
    action5, need_inference5 = executor.get_next_action()
    np.testing.assert_array_equal(action5, [30, 30])
    assert need_inference5


def test_reset():
    """Test reset clears all buffers."""
    executor = ActionChunkExecutor(chunk_horizon=4, overlap=2)
    chunk = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    executor.add_chunk(chunk)

    executor.get_next_action()
    executor.get_next_action()

    assert not executor.is_empty
    assert executor.remaining_actions == 2

    executor.reset()
    assert executor.is_empty
    assert executor.remaining_actions == 0


def test_invalid_chunk_shape():
    """Test validation of chunk dimensions."""
    executor = ActionChunkExecutor(chunk_horizon=4, overlap=2)

    # Wrong number of actions
    with pytest.raises(ValueError, match="Expected chunk size 4"):
        executor.add_chunk(np.array([[1, 2], [3, 4]], dtype=np.float32))

    # Wrong shape (1D)
    with pytest.raises(ValueError, match="Expected 2D action chunk"):
        executor.add_chunk(np.array([1, 2, 3, 4], dtype=np.float32))

    # Non-finite values
    with pytest.raises(ValueError, match="non-finite"):
        chunk = np.array([[1, 2], [np.inf, 4], [5, 6], [7, 8]], dtype=np.float32)
        executor.add_chunk(chunk)


def test_empty_buffer_error():
    """Test error when getting action from empty buffer."""
    executor = ActionChunkExecutor(chunk_horizon=4, overlap=2)

    with pytest.raises(RuntimeError, match="Action buffer is empty"):
        executor.get_next_action()


def test_invalid_configuration():
    """Test validation of executor configuration."""
    with pytest.raises(ValueError, match="chunk_horizon must be positive"):
        ActionChunkExecutor(chunk_horizon=0, overlap=1)

    with pytest.raises(ValueError, match="overlap must be in"):
        ActionChunkExecutor(chunk_horizon=4, overlap=0)

    with pytest.raises(ValueError, match="overlap must be in"):
        ActionChunkExecutor(chunk_horizon=4, overlap=4)


def test_stays_at_last_action():
    """Test that executor stays at last action if no new chunk provided."""
    executor = ActionChunkExecutor(chunk_horizon=4, overlap=2, blend_overlap=False)
    chunk = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    executor.add_chunk(chunk)

    # Consume all 4 actions
    for _ in range(4):
        executor.get_next_action()

    # Try to get one more (should repeat last action)
    action_repeat, _ = executor.get_next_action()
    np.testing.assert_array_equal(action_repeat, [4, 4])

    # Should keep returning last action
    action_repeat2, _ = executor.get_next_action()
    np.testing.assert_array_equal(action_repeat2, [4, 4])
