"""
## nanoEPics Jukebox
brownian_oscillator.py
classes representing the signal from an oscillator with Brownian noise

### tested classes:

+ damped_jiggler: overdamped oscilator, actuation frequency much lower than internal resonace
+ sawtooth_jiggler: forced linear response to a sawtooth signal of order n and adjustable time symmetry

### planned:

+ trapped_jiggler: Brownian particle inside an optical trap with forced oscillations


    .. lastedit:: 21/12/2021
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
import matplotlib.pyplot as plt

class sawtooth_jiggler:
    """
    Generates time-trace of signal from a sawtooth oscillator with added thermal noise. Here, the response is considered to be linear to the actuation force.

    Parameters
    ----------
    amplitude: sinusoidal oscillation amplitude
    freq: sinusoidal oscillation frequency in Hertz
    baseline: average signal
    noise: background random noise, in case of thermal noise Var(x) = kBT/kappa with kappa the internal spring constant
    order: number of higher harmonics considered
    phase: phase increment between successive harmonics, determines the sawtooth shape
    drift: average drift of the signal in time
    frate: time-series measurement rate = number of datapoints measured per second

    Returns
    -------
    float numpy array of signal(time)
    """

    def __init__(self, freq, amplitude, noise, order = 2, phase = 0, frate = 1000, baseline = 0, drift = 0):
        self.amplitude = amplitude
        self.freq = freq
        self.drift = drift
        self.frate = frate
        self.order = order,
        self.phase = phase,
        self.baseline = baseline
        self.noise = noise
        self.drift = drift

    def jig(self, duration):
        # duration: length of the time series in seconds
        n = int(duration * self.frate) # number of datapoints to be generated
        taxis = np.arange(n)/self.frate
        fsum = 0.0 * taxis
        o = self.order[0]
        for i in range(o):
            fsum = fsum + np.power(-1,i)*np.sin(2*np.pi*(i+1)*self.freq*taxis+self.phase)/(i+1)/np.pi
        nz = np.random.normal(size=n, loc=self.baseline, scale=self.noise)
        sig = self.amplitude * fsum + nz
        return sig, taxis

class damped_jiggler:
    """
    Generates time-trace of signal from a single oscillator with added thermal noise. For this type of oscillator we assume that the internal resonance frequency (corner frequency) is much higher than the actuation frequency, and therefore, the noise factor is constant independent of the measurement frate.

    Parameters
    ----------
    amplitude: sinusoidal oscillation amplitude
    freq: sinusoidal oscillation frequency in Hertz
    baseline: average signal
    noise: background random noise, in case of thermal noise Var(x) = kBT/kappa with kappa the internal spring constant
    frate: time-series measurement rate = number of datapoints per second
    drift: average drift of the signal in time


    Returns
    -------
    float numpy array of signal(time)
    """

    def __init__(self, freq, amplitude, noise, frate = 1000, baseline = 0  , drift = 0):
        self.amplitude = amplitude
        self.freq = freq
        self.drift = drift
        self.frate = frate
        self.baseline = baseline
        self.noise = noise
        self.drift = drift

    def jig(self, duration):
        # duration: length of the time series in seconds
        n = np.int(duration * self.frate) # number of datapoints to be generated
        taxis = np.arange(n)/self.frate
        nz = np.random.normal(size=n, loc=self.baseline, scale=self.noise)
        sig = self.amplitude * np.sin(2*np.pi*self.freq*taxis) + nz
        return sig, taxis


if __name__ == "__main__":
    #j = damped_jiggler(1, 1, 0.5, frate=100, baseline=20)
    j = sawtooth_jiggler(10, 1, 0.0, order = 4, phase = np.pi, frate=100, baseline=0)
    s, t = j.jig(1)   # creates at time series of xx seconds long with the parameters set in the class initiation
    plt.plot(t, s)
    plt.xlabel("time in seconds")
    plt.ylabel("amplitude")
    plt.show()
    
