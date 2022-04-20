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

plt.close('all')

fov = [160,230]
nf = 50
testframe = nf-1

#make a simulated dataset
meas = Landing_Flashes(seed = 314, fov=fov, numpar = 50, nframes = nf, signal = 15, sizevar=0.8, noise = 0, normalvar = False, dark = 10, psize = 2, unevenIllumination = False)
plis = meas.parlist
movie = meas.genStack()

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


def filmpje(moviedata):
    plt.figure()
    for i in range(nf):
        time.sleep(1./100)
        plt.imshow(moviedata[i,:,:])
        plt.show()
    return


#%%
#Mess around with the noise
plt.close('all')
meas.noise = 0
plis = meas.parlist
movie = meas.genStack()

plt.imshow(movie[-1,:,:])

plt.figure()
plt.title('Simulated Size histogram')
plt.hist(plis[:,2])
plt.savefig('GraphsandIms\\Simulated size histogram')

#%%
#Mess around with trackpy

f = trackpy.locate(movie[testframe], 5, minmass = 20, invert = False)
print(len(f['mass']))
trackpy.annotate(f, movie[testframe]);


#%%
#Run trackpy on the movie, with different levels of noise, make histograms
plt.close('all')

noises = [0,0.1,1,10]
masses = []
for i in range(len(noises)): 
    noise = noises[i]
    meas.noise = noise
    plis = meas.parlist
    movie = meas.genStack()

    
    f = trackpy.locate(movie[testframe], 5, minmass = 20, invert = False)
    print(len(f['mass']))

    masses.append(f['mass'])
     
    plt.figure()
    plt.title('Trackpy locate on last frame noise = '+ str(noise))
    trackpy.annotate(f, movie[testframe]);
    plt.savefig('GraphsandIms\\Trackpy_nobgcor_noise'+str(i))


fig, axs = plt.subplots(2,2)
fig.suptitle('Found Size Histograms using Trackpy and different noises')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.5)

axs[0,0].hist(masses[0], label = str(len(masses[0])))
axs[0, 0].set_title('Noise = ' + str(noises[0]))
axs[0,0].legend(prop={'size': 10})
axs[0,1].hist(masses[1],  label = str(len(masses[1])))
axs[0, 1].set_title('Noise = ' + str(noises[1]))
axs[0,1].legend(prop={'size': 10})
axs[1,0].hist(masses[2],  label = str(len(masses[2])))
axs[1, 0].set_title('Noise = ' + str(noises[2]))
axs[1,0].legend(prop={'size': 10})
axs[1,1].hist(masses[3],  label = str(len(masses[3])))
axs[1, 1].set_title('Noise = ' + str(noises[3]))
axs[1,1].legend(prop={'size': 10})


for ax in axs.flat:
    ax.set(xlabel='mass', ylabel='counts')

plt.savefig('GraphsandIms\\Trackpy_nocorrection_histograms')

#%%
#Do simple backgroundcorrection
#Simple mode: Take off the reference (= median), divide by reference - dark

def Simple_backgroundcor(mov):
    bg = np.median(mov,axis=0)

    backgroundcor = np.zeros((nf,mov[0].shape[0],mov[0].shape[1]))

    dark = np.ones((fov))*10
    for i in range(nf):
        backgroundcor[i] = (mov[i] - bg)/(bg-dark)

    #Set negative values to 0 
    backgroundcor[backgroundcor<0] = 0  
    
    return backgroundcor

#%%
#Again, mess around with trackpy

meas.noise = 10
plis = meas.parlist
movie = meas.genStack()

backgroundcorrected = Simple_backgroundcor(movie)

f = trackpy.locate(backgroundcorrected[testframe], 5, minmass = 20, invert = False)
print(len(f['mass']))
trackpy.annotate(f, backgroundcorrected[testframe]);


#%%
#Make histograms for 4 different levels of noise
plt.close('all')

noises = [0,0.1,1,10]
masses = []
for i in range(len(noises)): 
    noise = noises[i]
    plis = meas.parlist
    movie = meas.genStack()
    backgroundcor = Simple_backgroundcor(movie)
    
    testframe = nf-1
    
    f = trackpy.locate(backgroundcor[testframe], 15, minmass = 12, invert = False)
    print(len(f['mass']))

    masses.append(f['mass'])
     
    plt.figure()
    plt.title('Trackpy locate with backgroundcor on last frame noise = '+ str(noise))
    trackpy.annotate(f, movie[testframe]);
    plt.savefig('GraphsandIms\\Trackpy_withbgcor_noise'+str(i))
    
    plt.figure()
    plt.title('Backgroundcorrected last frame noise = ' + str(noise))
    plt.imshow(backgroundcor[testframe])


fig, axs = plt.subplots(2,2)
fig.suptitle('Found Size Histograms using Trackpy, different noises, simple backgroundcorrection')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.5)

axs[0,0].hist(masses[0], label = str(len(masses[0])))
axs[0, 0].set_title('Noise = ' + str(noises[0]))
axs[0,0].legend(prop={'size': 10})
axs[0,1].hist(masses[1],  label = str(len(masses[1])))
axs[0, 1].set_title('Noise = ' + str(noises[1]))
axs[0,1].legend(prop={'size': 10})
axs[1,0].hist(masses[2],  label = str(len(masses[2])))
axs[1, 0].set_title('Noise = ' + str(noises[2]))
axs[1,0].legend(prop={'size': 10})
axs[1,1].hist(masses[3],  label = str(len(masses[3])))
axs[1, 1].set_title('Noise = ' + str(noises[3]))
axs[1,1].legend(prop={'size': 10})


for ax in axs.flat:
    ax.set(xlabel='mass', ylabel='counts')

plt.savefig('GraphsandIms\\Trackpy_withcorrection_histograms')
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
