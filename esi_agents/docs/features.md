# Feature Library

The `esi_agents.features` module provides windowing utilities and domain-specific features for rotating machinery.

## Windowing

`generate_windows` produces sliding windows with configurable size and stride, preserving asset and channel identifiers.

## Time-domain

- RMS, standard deviation, peak-to-peak
- Kurtosis, crest factor
- Moving standard deviation and rolling z-score
- Linear trend slope and seasonal energy

## Frequency-domain

- FFT power and spectral centroid
- Band power across thirds of the spectrum
- Dominant frequency peaks

## Envelope & Demodulation

- Hilbert envelope mean, RMS, and peak
- Envelope spectrum peak frequency

## Order Tracking

- Harmonic amplitudes at 1×/2×/3× orders using RPM traces
- Sideband ratio around the dominant peak

All features are implemented with vectorised NumPy/Pandas operations and accept the `Window` objects produced by the windowing utilities.
