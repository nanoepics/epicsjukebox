"""
    epicsjukebox.signals.particle_drop.py
    ==================================
    This file contains verious classes for generating synthetic images corresponding to a suspension of particles passing the field of view. The simulation contains various noise sources, and keeps the ground truth for futher analyssi.
    These classes can be used for example to test the reliability of the particle counting algorithms

    .. lastedit:: 21/03/2022
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import time


class Kymograph:
    """Generates z-position vs time for a group of particles as they pass through the field of view with
    normal Brownian motion and drift.

    Parameters
    ----------
    fov: 1D field of view in pixels
    numpar: number of particles in the field of view,
    difcon: diffusion constant (identical particles) [pixel^2/frame]

    psize: particle size in pixels
    signal: brightness of each particle
    noise: background random noise
    drift: average drift velosity [pixel/frame]

    Returns
    -------
    float numpy array of intensity(position,time)
    """

    def __init__(self, fov = 500, numpar = 4, nframes = 30, difcon = 1, signal = 10, noise = 1, psize = 8, drift = 1):
        self.fov = fov
        self.difcon = difcon
        self.drift = drift
        self.numpar = numpar
        self.signal = signal
        self.noise = noise
        self.psize = psize
        self.nframes = nframes # number of lines (frames) to be generated
        self.tracks = np.zeros((numpar*nframes,5)) #array with all actual particle coordinates in the format [0-'tag', 1-'t', 2-'mass', 3-'z', 4-'width'] prior to adding noise

    def genKymograph(self):
        numpar = self.numpar
        nframes = self.nframes
        fov = self.fov
        positions = 0.8 * fov * (np.random.rand(numpar) + 0.1) # additional factors for making sure particles are generated not to close to the two ends
        kg = np.zeros((fov, nframes))
        taxis = np.arange(nframes)
        p_tag = 0
        for p in positions:  # generating random-walk assuming dt=1
            steps = np.random.standard_normal(self.nframes)
            path = p + np.cumsum(steps) * np.sqrt(2 * self.difcon) + self.drift * taxis
            intpath = np.mod(np.asarray(path, dtype=int), fov)
            kg[[intpath, taxis]] += self.signal * (1 + p_tag / 10)
            # nest few lines to fill in tracks in the format suitable for analysis
            p_tag += 1
            tags = np.array([((0*taxis)+1)*p_tag])
            masses = tags / p_tag * self.signal * (0.9 + p_tag / 10)
            widths = tags / p_tag * self.psize
            trackspart = np.concatenate((tags, [taxis], masses, [path], widths), axis=0)
            self.tracks[(p_tag-1)*nframes:p_tag*nframes,:] = np.transpose(trackspart)

        fft_tracks = np.fft.rfft2(kg, axes=(-2,))
        max_freq = int(self.fov / self.psize)
        fft_tracks[max_freq:, :] = 0
        kg = abs(np.fft.irfft2(fft_tracks, axes=(-2,)))
        noise = np.random.randn(self.fov, self.nframes)
        kg += noise
        return kg


class Landing_Flashes:
    """
    Generates a stack of images corresponding to landing of Brownian particles, assuming each particles lands in single frame (no time trace).

    Parameters
    ----------
    :fov: [width, height] of the desired image that contains these particles
    :bgframe: stationary background image given as input
    Returns
    -------
    :return: .loca: intended location of the particles (with sub-pixel resolution)
             .stack: a stack of images with specified noise and particles displaced accordingly
    """
    def __init__(self, fov = [300, 200], nframes=30, numpar = 20, signal = 10.0, sizevar = 0.3, noise = 10.0, bgframe = [None], dark = 100, psize = 6, unevenIllumination = False, irefmode = 2):
        # camera and monitor parameters
        self.xfov, self.yfov = fov
        self.numpar = numpar # Desired number of landing particles
        self.nframes = nframes # Desired number of recorded images
        self.signal = signal # brightness for each particle
        self.sizevar = sizevar # width of the size distribution assuming a square histogram
        self.noise = noise # background noise
        self.psize = psize # diameter of each particle in the image, currently must be integer

        if unevenIllumination:
            self.iref = self.genIref(irefmode)
        else:
            self.iref = np.ones(fov)

        self.psf = self.initPSF(psize)
        if bgframe == [None]:
            self.bg = self.genBG(dark, noise)
        else:
            self.bg = bgframe

        ts = np.random.uniform(0, numpar, size=(self.nframes, 1))
        self.nlanded = np.sort(ts.astype(int), axis=0)
        self.loca = self.initLocations()

    def genBG(self, dar, noi):
        """
        generates constant noisy background with extra dark signal
        """
        bg = np.random.poisson(noi, size = (self.xfov, self.yfov)) + dar
        ffbg = np.fft.rfft2(bg)
        max_freq = int(self.xfov / self.psize)
        ffbg[max_freq:, max_freq:] = 0
        bg = abs(np.fft.irfft2(ffbg))

        return bg

    def genIref(self, md):
        """
        generates uneven illumination pattern
        """
        cox, coy = np.meshgrid(np.arange(self.xfov)/self.xfov, np.arange(self.yfov)/self.yfov, indexing='ij')
        ir = self.noise*np.sin(2*np.pi*md*(cox+coy))+self.signal

        return ir
        
    def initPSF(self, p):
        psf = np.zeros((2*p,2*p))
        for n in range(p):
            for m in range(p):
                psf[p+n,p+m] = psf[p-n-1,p+m] = psf[p+n,p-m-1] = psf[p-n-1,p-m-1] = np.exp(-np.sqrt(n**2+m**2))
        return psf
        

    def initLocations(self):
        # initializes the random location of numpar particles in the frame. one can add more paramaters like intensity
        # and PSF distribution if necessary
        p = self.psize
        parx = np.random.uniform(2*p, self.xfov-2*p, size=(self.numpar, 1))
        pary = np.random.uniform(2*p, self.yfov-2*p, size=(self.numpar, 1))
        pari = np.random.uniform(self.signal*(1-self.sizevar/2), self.signal*(1+self.sizevar/2), size=(self.numpar, 1))

        lp = np.concatenate((parx, pary, pari), axis=1)

        return lp


    def genImage(self,npar):
        """
        :return: generated image with specified position in self.loca up to particle number n
        """
        simimage = np.copy(self.bg)
        psize = self.psize
        if npar > self.numpar:
            m = self.numpar
        else:
            m = int(npar)
        for n in range(m):
            x = int(self.loca[n,0])
            y = int(self.loca[n,1])
            simimage[x-psize:x+psize, y-psize:y+psize] = simimage[x-psize:x+psize, y-psize:y+psize] + self.psf * self.loca[n,2]

        simimage = np.multiply(simimage,self.iref)
        return simimage


    def genStack(self):
        """
        Using all the above methods in this class, this method only iterates enough to create a stack of synthetic frames
        that can be analyzed later

        :param nframes: number of frames to generate
        :return: simulated data
        """
        numpar = self.numpar
        nf = self.nframes
        data = np.zeros((self.xfov, self.yfov, nf))
        for n in range(nf):
            l = np.random.poisson(self.signal, size = (self.xfov, self.yfov))
            npar = self.nlanded[n]
            data[:,:,n] = np.multiply(self.genImage(npar), l)

        return data



nf = 10
meas = Landing_Flashes(fov=[160,230], numpar = 136, nframes = nf, signal = 20, sizevar=0.5, dark = 10, psize = 6, unevenIllumination = True)


sig = meas.genStack()
im = plt.imshow(sig[:,:,0])

for i in range(nf):
    plt.imshow(sig[:,:,i])
    time.sleep(0.1)
    plt.show()


