import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use a non-interactive backend for saving plots
from ipywidgets import interact, FloatSlider
import numpy as np
from src.utils.utilities import (detect_sudden_change_events, calculate_properties_possible_events)


def plot_data(data):
    plt.plot(data)
    plt.show()


def _plot_detected_events(window_size, pre_event_window, event_window, threshold, step,
                          pupil_diameter, pupil_times, pupil_sampling_rate):
    # Convert parameters from seconds to samples
    ws = int(window_size * pupil_sampling_rate)
    pev = int(pre_event_window * pupil_sampling_rate)
    ew = int(event_window * pupil_sampling_rate)
    st = int(step * pupil_sampling_rate)
    padding = 5

    # Detect events
    events, events_indices = detect_sudden_change_events(
        pupil_diameter,
        padding,
        pev,
        ew,
        threshold,
        st
    )

    # Build event indicator array
    event_or_not = np.zeros_like(pupil_diameter)
    for idx, event_idx in enumerate(events_indices):
        start = st * idx
        end = start + pev + ew
        if event_idx == 1:
            event_or_not[start:end] = 1
        elif event_idx == 2:
            event_or_not[start:end] = -1

    # Apply smoothing if requested
    if ws > 1:
        kernel = np.ones(ws) / ws
        pupil_smooth = np.convolve(pupil_diameter, kernel, mode='same')
    else:
        pupil_smooth = pupil_diameter

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.plot(pupil_times, pupil_diameter, label='Raw')
    plt.plot(pupil_times, pupil_smooth, label=f'Smoothed ({window_size}s)')
    plt.fill_between(pupil_times,
                     np.min(pupil_diameter),
                     np.max(pupil_diameter),
                     where=event_or_not > 0,
                     alpha=0.3,
                     label='Detected Event +')
    plt.fill_between(pupil_times,
                     np.min(pupil_diameter),
                     np.max(pupil_diameter),
                     where=event_or_not < 0,
                     alpha=0.3,
                     label='Detected Event -')
    plt.xlabel('Time (s)')
    plt.ylabel('Pupil Diameter')
    plt.title('Pupil Event Detection')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_detected_events_interactive(pupil_diameter, pupil_times, pupil_sampling_rate):
    """
    Launch an interactive widget for adjusting detection parameters.
    """
    interact(
        lambda window_size, pre_event_window, event_window, threshold, step:
            _plot_detected_events(
                window_size, pre_event_window, event_window, threshold, step,
                pupil_diameter, pupil_times, pupil_sampling_rate
            ),
        window_size=FloatSlider(value=1, min=0, max=60, step=1, description='Smoothing (s)'),
        pre_event_window=FloatSlider(value=5, min=1, max=20, step=0.5, description='Baseline (s)'),
        event_window=FloatSlider(value=5, min=1, max=20, step=0.5, description='Event Window (s)'),
        threshold=FloatSlider(value=3, min=2, max=10, step=0.5, description='Threshold (SD)'),
        step=FloatSlider(value=0.5, min=0.5, max=10, step=0.25, description='Step Size (s)')
    )


def find_best_events(
    block,
    pupil_diameter,
    time,
    whisker_time,
    whisker_velocity,
    print_result=True,
    plot_result=True,
    wakeup=True
):
    """
    Identify the optimal event within a candidate block, supporting wakeup (pupil dilation) or sleep (constriction).

    Args:
        block (tuple): (start_idx, end_idx) of candidate segment in pupil data.
        pupil_diameter (np.ndarray): Full pupil diameter signal.
        time (np.ndarray): Corresponding time points for pupil data.
        whisker_time (np.ndarray): Time points for whisker velocity.
        whisker_velocity (np.ndarray): Normalized whisker velocity signal.
        print_result (bool): If True, print metrics for filtered events.
        plot_result (bool): If True, overlay event lines and whisker trace.
        wakeup (bool): If True, apply wakeup criteria; else, apply sleep criteria.

    Returns:
        int or None: Absolute index of chosen event (in pupil samples), or None if none.
    """
    
    start, end = block
    pupil_seg = pupil_diameter[start:end]
    time_seg = time[start:end]

    # Compute candidate event properties, passing wakeup flag
    analysis_results = calculate_properties_possible_events(block, pupil_diameter, time, wakeup=wakeup)

    # Filter based on wakeup vs. sleep criteria
    if wakeup:
        filtered = [r for r in analysis_results if r[4] > r[2] * 3 and r[1] < 0.5 and (r[3] - r[1]) > 0.2]
    else:
        filtered = [r for r in analysis_results if r[4] > 0 and (r[1] - r[3]) > 0.3]

    # Reporting
    if print_result:
        for idx, base_mean, base_std, ev_mean, ev_std, downs, def_mag in filtered:
            print(f"Start idx: {idx}, Baseline mean={base_mean:.3f}, std={base_std:.3f}, "
                  f"Event mean={ev_mean:.3f}, std={ev_std:.3f}, downs={downs}, mag={def_mag:.3f}")

    # Visualization
    if plot_result:
        plt.plot(time_seg, pupil_seg, label='Pupil')
        for idx, *_ in filtered:
            plt.axvline(time_seg[idx], color='red', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Pupil')
        plt.title('Candidate Events')
        plt.legend()

        # Overlay whisker
        wi = np.searchsorted(whisker_time, time_seg[0])
        wf = np.searchsorted(whisker_time, time_seg[-1])
        plt.twinx().plot(whisker_time[wi:wf], whisker_velocity[wi:wf], label='Whisker', alpha=0.6)
        plt.ylabel('Whisker Vel')
        plt.show()

    # Choose optimal event by maximum deflection magnitude
    optimal = max(filtered, key=lambda x: x[6], default=None)

    if print_result:
        if optimal:
            print("\nOptimal Event:")
            idx, base_mean, base_std, ev_mean, ev_std, downs, def_mag = optimal
            print(f"Idx: {idx}, def_mag: {def_mag:.3f}")
        else:
            print("No optimal event found.")

    return (start + optimal[0]) if optimal else None