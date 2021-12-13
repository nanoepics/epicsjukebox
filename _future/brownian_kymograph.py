"""
    CETgraph.tracking.simulate.py
    ==================================
    Contains verious classes for generating synthetic images corresponding to particles performing a thermal Brownian
    motion to be viewed, and eventually analyzed when necessary.
    These classes can be used for example to test the reliability of the tracking algorithms

    .. lastedit:: 16/11/2017
    .. sectionauthor:: Sanli Faez <s.faez@uu.nl>
"""
import numpy as np

class Waterfall:
    def __init__(self):
        print("Class Waterfall has been renamed to class Kymograph.")

class Kymograph:
    """Generates z-position vs time for a group of particles as they go with
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

    def __init__(self, fov = 500, numpar = 4, nframes = 100, difcon = 1, signal = 10, noise = 1, psize = 8, drift = 1):
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
        wf[wf<self.signal/10] = 0  # to cancelt the artifacts of inverse FFT, for speckle background not correcting is more realistic
        kg += noise
        return kg
class SingleFrame:
    """
    :param size: [width, height] of the desired image that contains these particles
    :return: SingleFrame.loca: intended location of the particles (with sub-pixel resolution)
             SingleFrame.genImage: an image with specified noise and particles displaced accordingly
    """
    def __init__(self, fov = [300, 200], n = 4, difcon = 1, signal = 10, noise = 1, psize = 8):
        # camera and monitor parameters
        self.xfov, self.yfov = fov
        # simulation parameters
        self.difcon = difcon # Desired diffusion constant in pixel squared per frame
        self.numpar = n # Desired number of diffusing particles
        self.signal = signal # brightness for each particle
        self.noise = noise # background noise
        self.psize = psize # half-spread of each particle in the image, currently must be integer
        self.bg = self.genBG()
        self.initLocations()

    def initLocations(self):
        # initializes the random location of numpar particles in the frame. one can add more paramaters like intensity
        # and PSF distribution if necessary
        parx = np.random.uniform(0, self.xfov, size=(self.numpar, 1))
        pary = np.random.uniform(0, self.yfov, size=(self.numpar, 1))
        pari = np.random.uniform(1, self.numpar, size=(self.numpar, 1)) * self.signal
        self.loca = np.concatenate((parx, pary, pari), axis=1)
        self.loca = self.nextRandomStep()
        return self.loca

    def genBG(self):
        # generates a quite regular background image
        x = np.arange(self.xfov)
        y = np.arange(self.yfov)
        X, Y = np.meshgrid(y, x)
        simbg = 5*self.noise+np.sin((X+Y)/self.psize)
        return simbg

    def genImage(self):
        """
        :return: generated image with specified noise and particles position in self.loca
        """
        simimage = np.random.uniform(1, self.noise, size=(self.xfov, self.yfov)) + self.bg
        psize = self.psize
        normpar = np.zeros((2*psize, 2*psize))
        for x in range(psize):
            for y in range (psize):
                r = 2*(x**2+6*y**2)/psize**2
                normpar[psize-x, psize-y] = normpar[psize-x, psize+y] = \
                normpar[psize+x, psize-y] = normpar[psize+x, psize+y] = np.exp(-r)
        for n in range(0, self.numpar):
            x = np.int(self.loca[n,0])
            y = np.int(self.loca[n,1])
            simimage[x-psize:x+psize, y-psize:y+psize] = simimage[x-psize:x+psize, y-psize:y+psize] + normpar * self.loca[n,2]

        return simimage

    def nextRandomStep(self):
        numpar = self.numpar
        margin = 2*self.psize # margines for keeping whole particle spread inside the frame
        dr = np.random.normal(loc=0.0, scale=np.sqrt(2*self.difcon), size=(numpar, 2))
        locations = self.loca[:,0:2] + dr
        for n in range(0, numpar): # checking if particles get close to the margin
            for i in [0, 1]:
                if (locations[n, i] < margin) or (locations[n, i] > self.xfov-margin):
                    locations[n, i] = locations[n, i] - 2*dr[n, i] # applying mirror b.c.

        self.loca[:,0:2] = locations  #particle intensities are not varied between frames, only their locations
        simimage = self.genImage()
        return simimage

