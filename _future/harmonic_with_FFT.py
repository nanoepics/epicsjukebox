"""
## nanoEPics Jukebox
harmonic_with_FFT.py
class

### tested classes:


### planned:

+ bandpass: cleaning time-series
+ get_freq: extracting the main harmonic component of the signal

    .. lastedit:: 21/12/2021
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
import matplotlib.pyplot as plt

class bandpass:
    """Generates time-trace of signal from a single oscillator with added thermal noise. For this type of oscillator we assume that the internal resonance frequency (corner frequency) is much higher than the actuation frequency, and therefore, the noise factor is constant independent of the measurement frate.

    Parameters
    ----------
    amplitude: sinusoidal oscillation amplitude
    freq: sinusoidal oscillation frequency in Hertz
    baseline: average signal
    noise: background random noise, in case of thermal noise Var(x) = kBT/kappa with kappa the internal spring constant
    drift: average drift of the signal in time
    bandwidth: time-series measurement frate = number of datapoints per second

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