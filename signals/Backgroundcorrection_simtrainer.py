# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:19:23 2022

@author: Sjoerd Quaak
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from Particledrop import Landing_Flashes

import trackpy

np.random.seed(31415)
plt.close('all')

fov = [160,230]
nf = 50
meas = Landing_Flashes(fov=fov, numpar = 200, nframes = nf, signal = 15, sizevar=0.4, dark = 10, psize = 2, unevenIllumination = False)

plis = meas.parlist
print(np.shape(plis))
movie = meas.genStack()
#im = plt.imshow(movie[0,:,:])

#%%
def execute_amovie(moviedata):
    fig = plt.figure()
    axis = plt.axes()
    
    film = plt.imshow(moviedata[0])
    
    def init(): 
        film.set_data(moviedata[0])
        return [film]
    
    def animate(i):
        film.set_array(moviedata[i])
        return [film]
    
    anim = FuncAnimation(fig, animate, init_func = init, 
                         frames = moviedata.shape[0], interval = 20, blit = True)

    return anim


def filmpje():
    plt.figure()
    for i in range(nf):
        time.sleep(0.1)
        plt.imshow(movie[i,:,:])
        plt.show()
    return

#%%
    
#Simple mode: Take off the reference (= median), divide by reference - dark

bg = np.median(movie,axis=0)

backgroundcor = np.zeros((nf,movie[0].shape[0],movie[0].shape[1]))

dark = np.ones((fov))*100
for i in range(nf):
    backgroundcor[i] = (movie[i] - bg)/(bg-dark)

#Set negative values to 0 
backgroundcor[backgroundcor<0] = 0
#%%
plt.close('all')

plt.figure()
plt.title('Simulated Size histogram')
plt.hist(plis[:,2])
plt.show()

testframe = 49
plt.figure()
plt.imshow(backgroundcor[testframe])

f = trackpy.locate(backgroundcor[testframe], 11, minmass = 2, invert = False)

fig, ax = plt.subplots()
plt.title('Found Size Histogram')
ax.hist(f['mass'])
ax.set(xlabel='mass', ylabel='count');

plt.figure()
trackpy.annotate(f, backgroundcor[testframe]);




#%%
#Now, lets try a background correction according to Kevins setup
#The idea is to subtract the average of each pixel from the last 6 frames. This is a moving average??
#Or is it to subtract the average of the entire image?

start = 6
end = nf

bgcKevin = movie[start:end,:,:] - np.mean([movie[start-5:end-5,:,:],
                                          movie[start-4:end-4,:,:],
                                          movie[start-3:end-3,:,:],
                                          movie[start-2:end-2,:,:],
                                          movie[start-1:end-1,:,:]],
                                          axis = 0)

b = bgcKevin

#%%

# Run trackpy finder:
# Google trackpy for more information on trackpy

f = trackpy.batch(b, (15,31), minmass=2) 

t = trackpy.link_df(f[f['mass']==f['mass']], 3, memory=1)
t2 = trackpy.filter_stubs(t, 4)

#%%
# Filter landing candidates as described before (with the linear decay).
# Note: this is not very efficiently or clearly coded.

interestingpathslist = []
for i in t2.groupby('particle'):  # Filter paths that are too short:
    x = next(iter(i[1]['frame'].to_dict().values()))
    if ( i[1]["mass"][x] > 5 ):
        interestingpathslist.append(i[1])
        
masslist = []
locationslist = []
decaylist = []
timelist = []
for i in interestingpathslist:  # Take interesting data from still accepted paths:
    x = next(iter(i['frame'].to_dict().values()))
    masslist.append(i['mass'][x])
    locationslist.append([i['x'][x],i['y'][x]])
    decayrate = i['mass']/i['mass'][x]
    decaylist.append(decayrate)
    timelist.append(i['frame'][x])

acceptedparticles=[]
for i in range(len(masslist)):  # Filter for expected decay shape:
    ddict = decaylist[i]
    m = masslist[i]
    l = decaylist[i]
    d = []
    for j in ddict.to_dict().values():
        d.append(j)
    # Interesting particle landings decay as follows:
    if (d[1]-0.8)**2>0.01:
        continue
    if (d[2]-0.6)**2>0.01:
        continue
    if (d[3]-0.4)**2>0.01:
        continue
    acceptedparticles.append(i)

"""So 'acceptedparticles' is now a list of indices for the 
'masslist', 'locationslist', 'decaylist' and 'timelist'
wherin the information of these landings is mostly saved.
"""
