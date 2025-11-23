from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

ARM_LEFT_GROUP = (5, 7, 9)
ARM_RIGHT_GROUP = (6, 8, 10)
LEG_LEFT_GROUP = (11, 13, 15)
LEG_RIGHT_GROUP = (12, 14, 16)

SYMMETRIC_KEYPOINT_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 6),  # shoulders
    (7, 8),  # elbows
    (9, 10),  # wrists
    (11, 12),  # hips
    (13, 14),  # knees
    (15, 16),  # ankles
)


def _swap_group_points(
    keypoints: np.ndarray,
    left_group: Iterable[int],
    right_group: Iterable[int],
) -> None:
    left_group = tuple(left_group)
    right_group = tuple(right_group)
    for left_idx, right_idx in zip(left_group, right_group):
        if max(left_idx, right_idx) >= keypoints.shape[0]:
            continue
        keypoints[left_idx, :], keypoints[right_idx, :] = (
            keypoints[right_idx, :].copy(),
            keypoints[left_idx, :].copy(),
        )


def _linked_groups(left_idx: int, right_idx: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if left_idx in LEG_LEFT_GROUP and right_idx in LEG_RIGHT_GROUP:
        return LEG_LEFT_GROUP, LEG_RIGHT_GROUP
    if left_idx in LEG_RIGHT_GROUP and right_idx in LEG_LEFT_GROUP:
        return LEG_RIGHT_GROUP, LEG_LEFT_GROUP
    if left_idx in ARM_LEFT_GROUP and right_idx in ARM_RIGHT_GROUP:
        return ARM_LEFT_GROUP, ARM_RIGHT_GROUP
    if left_idx in ARM_RIGHT_GROUP and right_idx in ARM_LEFT_GROUP:
        return ARM_RIGHT_GROUP, ARM_LEFT_GROUP
    return (left_idx,), (right_idx,)


def _compute_orientation_sign(keypoints: np.ndarray) -> float:
    def _pair(idx_a: int, idx_b: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if max(idx_a, idx_b) >= keypoints.shape[0]:
            return None
        if keypoints.shape[1] < 3:
            return keypoints[idx_a, :2], keypoints[idx_b, :2]
        if keypoints[idx_a, 2] < 0.2 or keypoints[idx_b, 2] < 0.2:
            return None
        return keypoints[idx_a, :2], keypoints[idx_b, :2]

    pair = _pair(5, 6) or _pair(11, 12)
    if not pair:
        return 0.0
    left, right = pair
    delta_x = right[0] - left[0]
    if abs(delta_x) < 1e-3:
        return 0.0
    return float(np.sign(delta_x))


def _enforce_pose_orientation(
    curr_kp: np.ndarray,
    orientation_sign: float,
    symmetry_pairs: Iterable[tuple[int, int]],
) -> np.ndarray:
    adjusted = curr_kp.copy()
    if orientation_sign == 0.0:
        return adjusted

    key_groups = [
        (ARM_LEFT_GROUP, ARM_RIGHT_GROUP),
        (LEG_LEFT_GROUP, LEG_RIGHT_GROUP),
    ]
    for left_group, right_group in key_groups:
        left_idx = left_group[0]
        right_idx = right_group[0]
        if max(left_idx, right_idx) >= adjusted.shape[0]:
            continue
        left_x = adjusted[left_idx, 0]
        right_x = adjusted[right_idx, 0]
        if np.isnan(left_x) or np.isnan(right_x):
            continue
        if orientation_sign >= 0 and left_x > right_x:
            _swap_group_points(adjusted, left_group, right_group)
        elif orientation_sign < 0 and left_x < right_x:
            _swap_group_points(adjusted, left_group, right_group)
    return adjusted


def _enforce_left_right_consistency(
    prev_kp: np.ndarray,
    curr_kp: np.ndarray,
    symmetry_pairs: Iterable[tuple[int, int]],
) -> np.ndarray:
    adjusted = curr_kp.copy()
    if prev_kp is None:
        return adjusted

    for left_idx, right_idx in symmetry_pairs:
        if max(left_idx, right_idx) >= curr_kp.shape[0]:
            continue
        prev_left = prev_kp[left_idx, :2]
        prev_right = prev_kp[right_idx, :2]
        curr_left = adjusted[left_idx, :2]
        curr_right = adjusted[right_idx, :2]

        direct = np.linalg.norm(curr_left - prev_left) + np.linalg.norm(curr_right - prev_right)
        swapped = np.linalg.norm(curr_right - prev_left) + np.linalg.norm(curr_left - prev_right)
        if swapped + 1e-6 < direct:
            group_left, group_right = _linked_groups(left_idx, right_idx)
            _swap_group_points(adjusted, group_left, group_right)
    return adjusted


def _enforce_history_consistency(
    curr_kp: np.ndarray,
    history: list[np.ndarray],
    symmetry_pairs: Iterable[tuple[int, int]],
    visibility_threshold: float = 0.2,
) -> np.ndarray:
    adjusted = curr_kp.copy()

    def _history_mean(index: int) -> Optional[np.ndarray]:
        samples = []
        for past in history:
            if index >= past.shape[0]:
                continue
            if past.shape[1] >= 3 and past[index, 2] < visibility_threshold:
                continue
            samples.append(past[index, :2])
        if not samples:
            return None
        return np.mean(samples, axis=0)

    for left_idx, right_idx in symmetry_pairs:
        if max(left_idx, right_idx) >= adjusted.shape[0]:
            continue
        left_mean = _history_mean(left_idx)
        right_mean = _history_mean(right_idx)
        if left_mean is None or right_mean is None:
            continue
        left_curr = adjusted[left_idx, :2]
        right_curr = adjusted[right_idx, :2]
        if np.any(np.isnan(left_curr)) or np.any(np.isnan(right_curr)):
            continue
        direct = np.linalg.norm(left_curr - left_mean) + np.linalg.norm(
            right_curr - right_mean
        )
        swapped = np.linalg.norm(right_curr - left_mean) + np.linalg.norm(
            left_curr - right_mean
        )
        if swapped + 1e-6 < direct:
            group_left, group_right = _linked_groups(left_idx, right_idx)
            _swap_group_points(adjusted, group_left, group_right)
    return adjusted


def track_person_across_frames(
    keypoint_results: list,
    *,
    max_distance: float = 0.3,
    min_confidence: float = 0.25,
    symmetry_pairs: Iterable[tuple[int, int]] = SYMMETRIC_KEYPOINT_PAIRS,
    velocity_blend: float = 0.2,
    history_size: int = 3,
) -> list[np.ndarray]:
    """
    Track the same person across frames using bounding box IoU and keypoint similarity.
    
    Parameters
    ----------
    keypoint_results: list[KeypointResult]
        List of keypoint detection results for each frame.
    max_distance: float
        Maximum normalized distance (0-1) between person centers to consider as the same person.
    min_confidence: float
        Minimum confidence score to consider a detection valid.
    
    Returns
    -------
    list[np.ndarray]
        List of tracked keypoints for the main person, shape: (num_keypoints, 3) per frame.
    """
    if not keypoint_results:
        return []
    
    tracked_keypoints: list[np.ndarray] = []
    prev_center: Optional[np.ndarray] = None
    prev_keypoints: Optional[np.ndarray] = None
    prev_velocity: Optional[np.ndarray] = None
    prev_orientation: float = 0.0
    history: list[np.ndarray] = []
    
    for result in keypoint_results:
        if result.keypoints.size == 0:
            # No detection, use previous keypoints if available
            if prev_keypoints is not None:
                tracked_keypoints.append(prev_keypoints.copy())
            else:
                tracked_keypoints.append(np.full((17, 3), np.nan))
            continue
        
        # Filter by confidence
        valid_detections = []
        for idx in range(len(result.keypoints)):
            kp = result.keypoints[idx]
            score = result.scores[idx] if idx < len(result.scores) else 0.0
            if score >= min_confidence:
                # Compute center of keypoints (using visible points only)
                visible_mask = kp[:, 2] > 0.2  # visibility threshold
                if visible_mask.sum() > 0:
                    center = kp[visible_mask, :2].mean(axis=0)
                    valid_detections.append((idx, center, kp, score))
        
        if not valid_detections:
            # No valid detection, use previous
            if prev_keypoints is not None:
                tracked_keypoints.append(prev_keypoints.copy())
            else:
                tracked_keypoints.append(np.full((17, 3), np.nan))
            continue
        
        # If we have a previous person, find the closest one
        if prev_center is not None and prev_keypoints is not None:
            best_idx = None
            best_distance = float('inf')
            
            for idx, center, kp, score in valid_detections:
                # Compute distance between centers
                distance = np.linalg.norm(center - prev_center)
                
                # Also check keypoint similarity for visible points
                visible_prev = prev_keypoints[:, 2] > 0.2
                visible_curr = kp[:, 2] > 0.2
                visible_both = visible_prev & visible_curr
                
                if visible_both.sum() > 0:
                    kp_distance = np.mean(
                        np.linalg.norm(
                            kp[visible_both, :2] - prev_keypoints[visible_both, :2],
                            axis=1
                        )
                    )
                    # Combined distance metric
                    combined_distance = 0.6 * distance + 0.4 * kp_distance
                else:
                    combined_distance = distance
                
                if combined_distance < best_distance and combined_distance <= max_distance:
                    best_distance = combined_distance
                    best_idx = idx
            
            if best_idx is not None:
                # Found matching person
                _, _, selected_kp, _ = valid_detections[best_idx]
                selected = selected_kp.copy()
                orientation = _compute_orientation_sign(selected)
                if orientation == 0.0:
                    orientation = prev_orientation
                selected = _enforce_pose_orientation(selected, orientation, symmetry_pairs)
                if history:
                    selected = _enforce_history_consistency(selected, history, symmetry_pairs)
                if prev_keypoints is not None:
                    selected = _enforce_left_right_consistency(prev_keypoints, selected, symmetry_pairs)
                    if prev_velocity is not None and 0.0 < velocity_blend < 1.0:
                        predicted = prev_keypoints[:, :2] + prev_velocity
                        visible = selected[:, 2] > 0.2
                        selected[visible, :2] = (
                            (1 - velocity_blend) * selected[visible, :2]
                            + velocity_blend * predicted[visible]
                        )
                tracked_keypoints.append(selected.copy())
                prev_center = valid_detections[best_idx][1]
                if prev_keypoints is not None:
                    prev_velocity = selected[:, :2] - prev_keypoints[:, :2]
                else:
                    prev_velocity = np.zeros_like(selected[:, :2])
                prev_keypoints = selected
                prev_orientation = orientation or prev_orientation
                history.append(selected.copy())
                if len(history) > history_size:
                    history.pop(0)
            else:
                # No match found, use highest confidence detection
                best_detection = max(valid_detections, key=lambda x: x[3])
                _, center, kp, _ = best_detection
                selected = kp.copy()
                orientation = _compute_orientation_sign(selected)
                if orientation == 0.0:
                    orientation = prev_orientation
                selected = _enforce_pose_orientation(selected, orientation, symmetry_pairs)
                if history:
                    selected = _enforce_history_consistency(selected, history, symmetry_pairs)
                if prev_keypoints is not None:
                    selected = _enforce_left_right_consistency(prev_keypoints, selected, symmetry_pairs)
                tracked_keypoints.append(selected.copy())
                prev_center = center
                prev_velocity = (
                    selected[:, :2] - prev_keypoints[:, :2]
                    if prev_keypoints is not None
                    else np.zeros_like(selected[:, :2])
                )
                prev_keypoints = selected
                prev_orientation = orientation or prev_orientation
                history.append(selected.copy())
                if len(history) > history_size:
                    history.pop(0)
        else:
            # First frame, use highest confidence
            best_detection = max(valid_detections, key=lambda x: x[3])
            _, center, kp, _ = best_detection
            orientation = _compute_orientation_sign(kp)
            selected = _enforce_pose_orientation(kp, orientation, symmetry_pairs)
            tracked_keypoints.append(selected.copy())
            prev_center = center
            prev_keypoints = selected.copy()
            prev_velocity = np.zeros_like(selected[:, :2])
            prev_orientation = orientation
            history = [selected.copy()]
    
    return tracked_keypoints


def track_person_across_frames_simple(
    keypoint_results: list,
    *,
    max_distance: float = 0.4,
    min_confidence: float = 0.25,
) -> list[np.ndarray]:
    if not keypoint_results:
        return []

    tracked: list[np.ndarray] = []
    prev_center: Optional[np.ndarray] = None
    prev_keypoints: Optional[np.ndarray] = None

    for result in keypoint_results:
        if result.keypoints.size == 0:
            tracked.append(prev_keypoints.copy() if prev_keypoints is not None else np.full((17, 3), np.nan))
            continue

        valid = []
        for idx in range(len(result.keypoints)):
            kp = result.keypoints[idx]
            score = result.scores[idx] if idx < len(result.scores) else 0.0
            if score < min_confidence:
                continue
            visible = kp[:, 2] > 0.2
            if not np.any(visible):
                continue
            center = kp[visible, :2].mean(axis=0)
            valid.append((idx, center, kp, score))

        if not valid:
            tracked.append(prev_keypoints.copy() if prev_keypoints is not None else np.full((17, 3), np.nan))
            continue

        if prev_center is not None and prev_keypoints is not None:
            best_idx = None
            best_distance = float("inf")
            for idx, center, kp, score in valid:
                distance = np.linalg.norm(center - prev_center)
                if distance < best_distance and distance <= max_distance:
                    best_distance = distance
                    best_idx = idx
            if best_idx is not None:
                selected = valid[best_idx][2].copy()
            else:
                selected = max(valid, key=lambda x: x[3])[2].copy()
        else:
            selected = max(valid, key=lambda x: x[3])[2].copy()

        tracked.append(selected.copy())
        prev_center = selected[selected[:, 2] > 0.2, :2].mean(axis=0) if np.any(selected[:, 2] > 0.2) else None
        prev_keypoints = selected.copy()

    return tracked


def smooth_keypoints_temporal(
    keypoints_sequence: list[np.ndarray],
    *,
    window_size: int = 3,
    min_visible: float = 0.3,
    mode: str = "median",
    ema_alpha: float = 0.6,
) -> list[np.ndarray]:
    """
    Apply temporal smoothing to keypoints using a moving average filter.
    
    Parameters
    ----------
    keypoints_sequence: list[np.ndarray]
        List of keypoints arrays, shape: (num_keypoints, 3) per frame.
    window_size: int
        Size of the moving average window (must be odd).
    min_visible: float
        Minimum visibility threshold for a keypoint to be considered valid.
    
    Returns
    -------
    list[np.ndarray]
        Smoothed keypoints sequence.
    """
    if not keypoints_sequence:
        return []
    
    if mode not in {"median", "ema"}:
        raise ValueError("mode must be 'median' or 'ema'")
    
    if mode == "ema":
        ema_alpha = float(np.clip(ema_alpha, 0.05, 0.95))
        smoothed: list[np.ndarray] = []
        prev_smoothed: Optional[np.ndarray] = None
        for current in keypoints_sequence:
            if prev_smoothed is None:
                smoothed_kp = current.copy()
            else:
                smoothed_kp = prev_smoothed.copy()
                visible = current[:, 2] >= min_visible
                smoothed_kp[visible, :2] = (
                    ema_alpha * current[visible, :2]
                    + (1 - ema_alpha) * prev_smoothed[visible, :2]
                )
                smoothed_kp[:, 2] = current[:, 2]
            smoothed.append(smoothed_kp)
            prev_smoothed = smoothed_kp
        return smoothed
    
    if window_size < 1:
        return keypoints_sequence
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed: list[np.ndarray] = []
    
    for i in range(len(keypoints_sequence)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(keypoints_sequence), i + half_window + 1)
        window_kps = keypoints_sequence[start_idx:end_idx]
        
        smoothed_kp = keypoints_sequence[i].copy()
        for kp_idx in range(smoothed_kp.shape[0]):
            valid_positions = []
            for kp_frame in window_kps:
                if kp_frame[kp_idx, 2] >= min_visible:
                    valid_positions.append(kp_frame[kp_idx, :2])
            if len(valid_positions) > 0:
                valid_array = np.array(valid_positions)
                smoothed_kp[kp_idx, :2] = np.median(valid_array, axis=0)
                smoothed_kp[kp_idx, 2] = keypoints_sequence[i][kp_idx, 2]
        smoothed.append(smoothed_kp)
    
    return smoothed

