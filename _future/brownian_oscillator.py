"""
nanoEPics Jukebox
brownian_oscillator.py
class representing the signal from an oscillator with Brownian noise

    .. lastedit:: 13/12/2021
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
import matplotlib.pyplot as plt

class damped_jiggler:
    """Generates time-trace of signal from a single oscillator with added thermal noise. For this type of oscillator we assume that the internal resonance frequency (corner frequency) is much higher than the actuation frequency, and therefore, the noise factor is constant independent of the measurement bandwidth.

    Parameters
    ----------
    amplitude: sinusoidal oscillation amplitude
    freq: sinusoidal oscillation frequency in Hertz
    baseline: average signal
    noise: background random noise
    drift: average drift of signal in time
    bandwidth: time-series measurement bandwidth = number of datapoints per second

    Returns
    -------
    float numpy array of signal(time)
    """

    def __init__(self, amplitude, freq, noise, bandwidth = 1000, baseline = 0  , drift = 0):
        self.amplitude = amplitude
        self.freq = freq
        self.drift = drift
        self.bandwidth = bandwidth
        self.baseline = baseline
        self.noise = noise
        self.drift = drift

    def jig(self, duration):
        # duration: length of the time series in seconds
        n = np.int(duration * self.bandwidth) # number of datapoints to be generated
        taxis = np.arange(n)/self.bandwidth
        nz = np.random.normal(size=n, loc=self.baseline, scale=self.noise)
        sig = self.amplitude * np.sin(2*np.pi*self.freq*taxis) + nz
        return sig, taxis


if __name__ == "__main__":
    j = damped_jiggler(1, 1, 0.5, bandwidth=100, baseline=20)
    s, t = j.jig(20)
    plt.plot(t, s)
    plt.show()