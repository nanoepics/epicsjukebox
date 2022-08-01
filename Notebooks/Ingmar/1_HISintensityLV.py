# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:19:09 2022

@author: Sjoerd Quaak

The aim of this script is to read the HIS files, play the movie and make an intensity plot
"""

#####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from readHISbasic import readSection
no = 24
# filedir = 'D:\\20220405\\'
filedir = r"C:\Users\brugg\OneDrive\Documenten\MasterThesis\Ingmar\20000001.HIS"
# fn = filedir + 'C'+str(no).zfill(5)+'.his'
fn = filedir



nframes = 205

#%%

#First, lets plot a single frame
m = np.memmap(fn, shape=None, mode = 'r')
offset = 0
img = readSection(m, offset)
plt.figure()

imgplot = plt.imshow(img)


#%%
#Now, let's make a movie dataset
movie = np.zeros((nframes,img.shape[0],img.shape[1]), dtype=np.uint16)
m = np.memmap(fn, shape=None, mode = 'r')
offset = 0

offsets = []
for i in range(nframes):
    offsets.append(offset)
    frame = readSection(m, offset)
    temp = np.copy(frame)
   #print(i, offset, temp.shape)
   
    movie[i] = np.copy(frame)
    offset = frame.HIS.offsetNext
    
 #Set values below a certain treshold to zero      
#movie[movie<450] = 0

    
#%%
#Tweak the last numbers to cut off more or less from the image, to get to the region of interest
#the first frame (called img) is used for this
    

border_ver1 =0 + 15
border_ver2 = img.shape[0] - 15
border_hor1 = 0 + 00
border_hor2 = img.shape[1]-100


h = img.shape[0]
w = img.shape[1]



img_cut = img[border_ver1:border_ver2,border_hor1:border_hor2]
plt.figure()
imgcutplot = plt.imshow(img_cut)



#%%
#Play the movie as an animation
#Call the function in the console
def execute_movie():
    fig = plt.figure()
    axis = plt.axes()
    
    film = plt.imshow(movie[0])
    
    def init(): 
        film.set_data(movie[0])
        return [film]
    
    
    def animate(i):
        film.set_array(movie[i])
        return [film]
    fig.suptitle('Measurement '+str(no))
    
    anim = FuncAnimation(fig, animate, init_func = init, 
                         frames = nframes, interval = 20, blit = True)

    return anim


#%%



#Or play the movie with the borders cut:
#Call the function in the console
def execute_movie_cut():

    fig = plt.figure()  
    axis = plt.axes()

    film = plt.imshow(movie[0][border_ver1:border_ver2,border_hor1:border_hor2])

    def init(): 
        film.set_data(movie[0][border_ver1:border_ver2,border_hor1:border_hor2])
        return [film]

    def animate(i):
        film.set_array(movie[i][border_ver1:border_ver2,border_hor1:border_hor2])
        return [film]
    
    fig.suptitle('Measurement cut '+str(no))
   
    
    anim = FuncAnimation(fig, animate, init_func = init,
                         frames = nframes, interval = 20, blit = True)

    
    return anim

execute_movie()

#%%
#
# plt.figure(figsize=(10,10))
# im=plt.imshow(movie[0, :,:])
# plt.colorbar()
# every_Nth_frame = 1
# for i in np.arange(1, movie.shape[0], every_Nth_frame):
#     im.set_data(movie[i, :,:])
#     plt.title(f'{i/20} s')
#     plt.pause(1./1000000)
#     #plt.savefig(f"D:\Anna\\20220703\Video_bg\Landing_Au_100nm_bgcorrected__{i:04}.jpeg")
# plt.show()
# #%%
#
# #plt.close('all')
# intensity = np.zeros(len(movie))
# minute_list = np.linspace(0,60,len(movie))
#
# for i in range(nframes):
#     intensity[i] = sum(sum(movie[i][border_ver1:border_ver2,border_hor1:border_hor2]))
# I=intensity/1000
# plt.figure()
# plt.plot(minute_list, I, label = no)
# plt.xlabel("Time (s)", fontsize=18)
# plt.xlim(0,60)
# plt.ylabel("Integrated Intensity (a.u.)", fontsize=18)
# plt.suptitle(str(no))
# plt.savefig('Intensity Measurement - number: '+str(no)+'.png')
# np.savetxt('intensity'+str(no)+'.csv', intensity, delimiter=',')
#
#
#
