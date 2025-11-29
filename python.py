import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
q0 = 0.5   # allocation to control
q1 = 0.5   # allocation to treatment
HR = 0.6   # hazard ratio
total_events_target = 121  # planned events
alpha = 0.05

# Z values
z_alpha = norm.ppf(1 - alpha / 2)
z_beta = norm.ppf(0.80)

# Event counts to explore
event_counts = np.arange(50, 300, 1)
powers = []

# Calculate power using Schoenfeld formula
for events in event_counts:
    se = 1 / np.sqrt(events * q0 * q1)
    z_effect = np.log(HR) / se
    power = 1 - (norm.cdf(z_effect - z_alpha) + (1 - norm.cdf(-z_effect - z_alpha)))
    powers.append(power * 100)

# Required events for 80% power
required_events = ((z_alpha + z_beta) ** 2) / ((np.log(HR)) ** 2 * q0 * q1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(event_counts, powers, color='blue', label=f'Power Curve (HR = {HR})')
plt.axhline(80, color='red', linestyle='--', label='80% Power Threshold')
plt.axvline(required_events, color='purple', linestyle='-.',
            label=f'Required Events ≈ {int(np.ceil(required_events))}')
plt.axvline(total_events_target, color='green', linestyle=':',
            label=f'Planned Events = {total_events_target}')

plt.xlabel('Total Number of Events')
plt.ylabel('Statistical Power (%)')
plt.title('Power Curve for Cox Model (HR = 0.6)')
plt.grid(True)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
