import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque


#Wrapper that abstracts the memory/frame pre-processing to simplify main code

class processingWrapper(object):
    def __init__(self, env, memory_length,h,w):
        #intializes the environment.
        #Takes an environment, length of the memory deque, and dimensions(h,w) of resized frame
        #Creates actions from action space, A

        self.env = env

        self.w = w
        self.h = h


        self.A = range(env.action_space.n)


        self.memory_length = memory_length
        self.memory = deque()


    def processFrame(self, frame):
        """
        mnih et all references preprocessing in 5.1, Experiemental Setup
        implementing similar system as mentioned in Mnih et al 2015
        
        Greyscale -> crop image

        Purpose is to remove computational complexity without removing information
        """
        return resize(rgb2gray(frame), (self.w, self.h))

    def initState(self):

        #memory deque for storing experiences
        self.memory = deque()

        x = self.env.reset()
        x = self.processFrame(x)

        #stack of states. Action repeat is 4 as specified in Mni et all 5.1
        s_t = np.stack((x, x, x, x), axis = 0)
        
        for i in range(self.memory_length-1):
            self.memory.append(x)
        return s_t



    def step(self, a):
 
        #The agent makes a move, a
        #Builds state deque with memories and with new frame
        #Returns new state information.


        #The environment variables, "info" is not used, so its not stored

        x_t1, r_t, done , _ = self.env.step(self.A[a])

        x_t1 = self.processFrame(x_t1)

        past = np.array(self.memory)

        s_t1 = np.empty((self.memory_length, self.h, self.w))

        #set past frames to the frames loaded from memory

        s_t1[:self.memory_length-1, ...] = past
        
        #newest frame is at front
        s_t1[self.memory_length-1] = x_t1

        # Forget the previous memory, add the current frame to memories queue

        self.memory.popleft()

        self.memory.append(x_t1)

        return s_t1, r_t, done
