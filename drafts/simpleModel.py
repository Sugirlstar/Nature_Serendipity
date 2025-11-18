# -*- coding: utf-8 -*-
"""
Model 1: AR(1) red noise with tau = 2 days
Model 2: Model 1 + 25-day sinusoidal oscillation
- Both models are mean-centered (zero mean)
- Same threshold 'a' is applied: if series > a for >=1 day, count as a blocking (blk) event
- Figure 1: Frequency domain (power spectrum) of both models
- Figure 2: Histogram of blocking-event durations (log y-axis), comparison of the two models
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
N = 50000                 # time length (days)
tau = 2.0                # e-folding time (days)
phi = np.exp(-1.0 / tau) # AR(1) coefficient (with daily step)
target_var = 1.0         # desired stationary variance Var(X)
sigma_eps = np.sqrt(target_var * (1 - phi**2))  # innovation noise std
period = 25.0            # sinusoidal period (days)
A = 0.8                  # sinusoidal amplitude
a = 1.5                  # blocking threshold (same for both models)

rng = np.random.default_rng(123)

# -----------------------
# Model 1: pure red-noise AR(1)
# -----------------------
x1 = np.zeros(N)
eps = rng.normal(0.0, sigma_eps, size=N)
for t in range(1, N):
    x1[t] = phi * x1[t-1] + eps[t]
x1 = x1 - np.mean(x1)  # force zero mean

# -----------------------
# Model 2: red-noise + 25-day oscillation
# -----------------------
t_idx = np.arange(N)
osc = A * np.cos(2 * np.pi * t_idx / period)   # sinusoidal term
x2 = x1 + osc
x2 = x2 - np.mean(x2)  # force zero mean

# -----------------------
# Simple periodogram (power spectrum)
# -----------------------
def periodogram(x, dt=1.0):
    n = len(x)
    X = np.fft.rfft(x)
    power = (np.abs(X)**2) / n            # power spectrum
    freqs = np.fft.rfftfreq(n, d=dt)      # frequencies (cycles/day)
    return freqs, power

f1, p1 = periodogram(x1)
f2, p2 = periodogram(x2)

# remove zero frequency
mask1 = f1 > 0
mask2 = f2 > 0

# -----------------------
# Blocking event detection
# -----------------------
def run_lengths_above(series, thr):
    """Return lengths of consecutive runs where series > threshold."""
    above = series > thr
    lens, run = [], 0
    for v in above:
        if v:
            run += 1
        else:
            if run > 0:
                lens.append(run)
                run = 0
    if run > 0:
        lens.append(run)
    return np.array(lens, dtype=int)

lens1 = run_lengths_above(x1, a)
lens2 = run_lengths_above(x2, a)

# Histogram of event durations
K = max(lens1.max() if len(lens1) else 1, lens2.max() if len(lens2) else 1)
bins = np.arange(1, K + 2)  # bins for durations 1,2,3,...
counts1, _ = np.histogram(lens1, bins=bins)
counts2, _ = np.histogram(lens2, bins=bins)
centers = np.arange(1, K + 1)

# -----------------------
# Figure 1: Frequency domain
# -----------------------
plt.figure(figsize=(9, 5.5))
plt.loglog(f2[mask2], p2[mask2], label="Model 2: red noise + 25-day oscillation", color="orange", alpha=0.7)
plt.loglog(f1[mask1], p1[mask1], label="Model 1: red noise", color="blue", alpha=0.7)
plt.xlabel("Frequency (cycles/day)")
plt.ylabel("Power")
plt.title("Frequency domain comparison")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('RedNoise_spectrum_comparison.png', dpi=300)

# -----------------------
# Figure 2: Histogram of event durations (log y-axis)
# -----------------------
plt.figure(figsize=(9, 5.5))
# Bar plots for the two models
plt.bar(centers - 0.2, counts1, width=0.4, alpha=0.7, color="steelblue", label="Model 1")
plt.bar(centers + 0.2, counts2, width=0.4, alpha=0.7, color="indianred", label="Model 2")
# Set log scale (base e)
plt.yscale("log", base=np.e)
plt.xlabel("Blocking-event duration (days)")
plt.ylabel("Number of events (log scale, base e)")
plt.title(f"Histogram of blocking-event durations (threshold a = {a})")
plt.legend()
plt.tight_layout()
# Save before showing
plt.savefig("RedNoise_duration_histogram.png", dpi=300)
plt.show()
plt.savefig('RedNoise_duration_histogram.png', dpi=300)

# -----------------------
# Print basic stats
# -----------------------
print("Model 1 mean/std:", float(np.mean(x1)), float(np.std(x1)))
print("Model 2 mean/std:", float(np.mean(x2)), float(np.std(x2)))
print("Threshold a:", a)
print("Model 1 total events:", int(counts1.sum()), "max duration:", int(lens1.max() if len(lens1) else 0))
print("Model 2 total events:", int(counts2.sum()), "max duration:", int(lens2.max() if len(lens2) else 0))

print('done')
