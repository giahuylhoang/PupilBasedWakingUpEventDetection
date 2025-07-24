import numpy as np
from src.utils.utilities import (
    detect_sudden_change_events, fill_false_between_trues,
    find_consecutive_true_blocks, check_cross_midline, calculate_derivative, normalize_series
)
from src.visualization.plotter import find_best_events

def detect_events(
    normalized_smoothed_pupil_size,
    smoothed_time_series,
    normalized_whisker_velocity,
    whisker_velocity_time,
    pupil_sampling_rate,
    whisker_sampling_rate,
    bsline_length,
    event_length,
    wakeup=False,
    plot_result=False
):
    """
    Detect wakeup or sleep events based on sudden changes in pupil size and whisker velocity.

    Args:
        normalized_smoothed_pupil_size (np.ndarray): Normalized and smoothed pupil size.
        smoothed_time_series (np.ndarray): Corresponding time points for pupil data.
        normalized_whisker_velocity (np.ndarray): Normalized whisker velocity.
        whisker_velocity_time (np.ndarray): Corresponding time points for whisker data.
        pupil_sampling_rate (float): Samples per second for pupil data.
        whisker_sampling_rate (float): Samples per second for whisker data.
        bsline_length (float): Baseline window length in seconds.
        event_length (float): Event window length in seconds.
        wakeup (bool): If True, detect wakeup events (increases); if False, detect sleep events (decreases).

    Returns:
        List of detected event indices (in pupil sample units) for either wakeup or sleep.
    """
    # Convert seconds to samples
    pre_event_window = int(bsline_length * pupil_sampling_rate)
    event_window = int(event_length * pupil_sampling_rate)

    # Detect sudden changes: increase (1) or decrease (2)
    events, _ = detect_sudden_change_events(
        normalized_smoothed_pupil_size,
        padding=None,
        pre_event_window=pre_event_window,
        event_window=event_window,
        threshold=2,
        step=1
    )

    # Build mask array marking event windows
    mask = np.zeros_like(normalized_smoothed_pupil_size, dtype=float)
    for start_idx, change_type in events:
        end_idx = start_idx + pre_event_window + event_window
        if end_idx > len(mask):
            end_idx = len(mask)
        if change_type == 'increase':
            mask[start_idx:end_idx] = 1
        elif change_type == 'decrease':
            mask[start_idx:end_idx] = -1

    # Select sleep (decrease) or wakeup (increase)
    if wakeup:
        event_mask = mask > 0.5
    else:
        event_mask = mask < -0.5

    # Fill small gaps
    filled_mask = fill_false_between_trues(event_mask, threshold=int(10 * pupil_sampling_rate))

    # Identify continuous blocks
    consecutive_blocks = find_consecutive_true_blocks(filled_mask, pupil_sampling_rate)

    # Filter blocks that cross midline in pupil size
    cross_midline_blocks = [
        (s, e) for s, e in consecutive_blocks
        if check_cross_midline(normalized_smoothed_pupil_size[s:e])
    ]

    # Further filter by amplitude change
    valid_blocks = []
    for s, e in cross_midline_blocks:
        amp = normalized_smoothed_pupil_size[s:e]
        if np.max(amp) - np.min(amp) > 0.5:
            valid_blocks.append((s, e))

    # Find best events and compute whisker integrals
    final_events = []
    integral_data = []
    for s, e in valid_blocks:
        idx = find_best_events(
            (s, e),
            normalized_smoothed_pupil_size,
            smoothed_time_series,
            whisker_velocity_time,
            normalized_whisker_velocity,
            print_result=False,
            plot_result=plot_result,
            wakeup=wakeup
        )
        if idx is not None:
            final_events.append(idx)
            # compute whisker integral ratio
            time_pt = smoothed_time_series[idx]
            w_idx = np.searchsorted(whisker_velocity_time, time_pt)
            base = normalized_whisker_velocity[w_idx - int(bsline_length * whisker_sampling_rate):w_idx]
            wake = normalized_whisker_velocity[w_idx:w_idx + int(event_length * whisker_sampling_rate)]
            integral_base = np.trapz(base)
            integral_wake = np.trapz(wake)
            integral_data.append(integral_wake / (integral_base + 1e-6))

    # For wakeup events, require ratio >1.5; for sleep events, ratio <0.67
    threshold_ratio = 1.5 if wakeup else 0.67
    selected = [evt for evt, r in zip(final_events, integral_data) if (r > threshold_ratio if wakeup else True)]
    selected = [evt for evt, r in zip(final_events, integral_data)]

    print(f"Selected {len(selected)} events after filtering by integral ratio.")
    if not selected:
        print("No events selected after filtering.")
        return []
    print(f"Selected events: {selected}")

    return selected
