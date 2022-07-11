import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PendulumEnv import pend_Hybrid #Debugged
import matplotlib.pyplot as plt 
import time 
from random import sample
import pandas as pd 
from numpy.random import randint, uniform
import hyper as h
import keyboard
from collections import deque

from tensorflow.python.ops.numpy_ops import np_config



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np_config.enable_numpy_behavior()




class DeepQNN:
    ''' This class depicts a Deep Q Network applied to a pendulum with a variable number of joints '''
    
    def __init__(self, nu, NN_h):
        self.nu = nu
        self.Joints = 1  # change this to 2 to test double pendulum 
        self.env = pend_Hybrid.HybridP(self.Joints, self.nu, dt=0.1)
        self.nx = self.env.nx
        
        self.printF = 25 # printing frequency
        self.export = True # flag to export cost to go into external pandas file 
        self.plotsFinal = True # flag to plot final costs to go 
        
        if NN_h == 4:
            self.q = self.get_critic4()
            self.q.summary()
        
            self.q_target = self.get_critic4()
            self.q_target.set_weights(self.q.get_weights())
            self.NN_layers = 4 
        elif NN_h == 3:
            self.q = self.get_critic3()
            self.q.summary()
        
            self.q_target = self.get_critic3()
            self.q_target.set_weights(self.q.get_weights())
            self.NN_layers = 3
        else:
            raise Exception("Sorry unavailable hidden layers architecture. choose between 3 or 4 ")
            exit()
        
        self.optimizer = tf.keras.optimizers.Adam(h.QVALUE_LEARNING_RATE)
        self.replay_buffer = deque(maxlen=h.REPLAY_BUFFER_SIZE)
        
        self.ctgRecord = []
        self.Bestc2go = np.inf

        self.cost2goImprov = []
        self.epochC2G = []

    

    def update(self, batch):

        ''' The weights of the NN are updated using the batch of data provided '''

        x_batch      = np.array([sample[0] for sample in batch])
        u_batch      = np.array([sample[1] for sample in batch])
        cost_batch   = np.array([sample[2] for sample in batch])
        x_next_batch = np.array([sample[3] for sample in batch])
        finished_batch   = np.array([sample[4] for sample in batch])
        
        n = len(batch)
        
        with tf.GradientTape() as tape:
            # Compute Q target
            target_output = self.q_target(x_next_batch, training=True).reshape((n,-1,self.Joints))
            target_value  = tf.math.reduce_sum(np.min(target_output, axis=1), axis=1)
            
            # Compute 1-step targets for the critic loss
            y = np.zeros(n)
            for id, finished in enumerate(finished_batch):
                if finished:
                    y[id] = cost_batch[id]
                else:
                    y[id] = cost_batch[id] + h.GAMMA*target_value[id]      
            
            # Compute Q
            qOut = self.q(x_batch, training=True).reshape((n,-1,self.Joints))
            a = np.repeat(np.arange(n),self.Joints).reshape(n,-1)
            b = u_batch.reshape(n,-1)
            c = np.repeat(np.arange(self.Joints).reshape(1,-1),n,axis=0)
            qVal  = tf.math.reduce_sum(qOut[a, b, c], axis=1)
            
            # Compute the loss
            loss = tf.math.reduce_mean(tf.math.square(y - qVal))

        # Apply gradient 
        DeltaQ = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(DeltaQ, self.q.trainable_variables))

    def chooseU(self, x, epsilon):
        ''' Choose the discrete control of the system following an epsilon greedy strategy'''
        if uniform(0,1) < epsilon:
            u = randint(self.nu, size=self.Joints)
        else:
            pred = self.q.predict(x.reshape(1,-1))
            u = np.argmin(pred.reshape(self.nu,self.Joints), axis=0)
    
        if len(u) == 1:
            u = u[0]
        return u

    def trainNN(self):
        ''' Training the NN '''
        
        
        steps = 0
        epsilon = h.EPSILON # import epsilon 
        
        t = time.time() # start measuring the time needed for training 
        
        for epoch in range(h.EPOCHS):
            c2go = 0.0
            x = self.env.reset()
            gamma = 1
            self.epochExport = epoch # keep track of current epoch for plotting on-the-go and exporting data to external .csv file 
            
            for step in range(h.MAX_EPOCH_LENGTH):
                u = self.chooseU(x, epsilon)
                x_next, cost = self.env.step(u)
                
                finished = True if step == h.MAX_EPOCH_LENGTH - 1 else False
                self.replay_buffer.append([x, u, cost, x_next, finished])
                
                # update weights of target network according to hyperparameters 
                if steps % h.UPDATE_Q_TARGET_STEPS == 0:
                    self.q_target.set_weights(self.q.get_weights())
                
                # Sampling from replay buffer and train NN accordingly 
                if len(self.replay_buffer) > h.MIN_BUFFER_SIZE and steps % h.SAMPLING_STEPS == 0:
                    batch = sample(self.replay_buffer, h.BATCH_SIZE)
                    self.update(batch)
                
                # update state , steps and hyperparameters accordingly
                x = x_next 
                c2go += gamma * cost
                gamma *= h.GAMMA
                steps += 1
            
            # Save NN weights everytime a better cost to go is found and plot the average cost to go vs the best 
            if abs(c2go) < abs(self.Bestc2go):
                self.saveModel()
                self.Bestc2go = c2go  # update best cost to go 
                self.cost2goImprov.append(self.Bestc2go) # store best cost to go 
                self.epochC2G.append(epoch)     # store at what epoch it was found 
                self.plotting(epoch)

            
            epsilon = max(h.MIN_EPSILON, np.exp(-h.EPSILON_DECAY*epoch)) # calculate the decay of epsilon
            self.ctgRecord.append(c2go) # append current cost to go in array 
            
            # Regularly print in terminal useful info about how the training is proceding 
            if epoch != 0 and epoch % self.printF == 0:
                dt = time.time() - t # calculate elapsed time since last printing to terminal 
                t = time.time()

                if epoch % 50 == 0: # save model every 50 epochs 
                    print(50*"--")
                    print("saving and plotting model at epoch", epoch)
                    self.saveModel()
                    #self.plotting(self.epochExport)

                print(50*"--")
                print('Epoch %d | cost %.1f | exploration prob epsilon %.6f | time elapase [s] %.5f s | cost to go improved in total %d times | best cost to go %.3f' % (
                      epoch, np.mean(self.ctgRecord[-self.printF:]), epsilon, dt, len(self.cost2goImprov), self.Bestc2go))
                print(50*"--")
                self.plotting(self.epochExport+1)
                
        if self.plotsFinal:
            self.plotting(h.EPOCHS)
            self.exportCosts(h.EPOCHS)
    
    def plotting(self, epochs):
        ''' Plot the average cost-to-go history and best cost to go and its relative epoch of when it was found  '''
        plt.plot(np.cumsum(self.ctgRecord)/range(1,epochs+1), color= 'blue')
        plt.scatter(self.epochC2G, self.cost2goImprov, color = 'red')
        plt.grid(True)
        plt.xlabel("epochs [n]")
        plt.ylabel("cost to go ")
        plt.legend(["Avg", "Best"])
        plt.title ("Average cost-to-go vs Best cost to go update")
        plt.savefig("costToGo.eps")
        #plt.show()
        #time.sleep(2)
        plt.close()

        #Epochcost = np.cumsum(self.ctgRecord)/range(1,int(epochs+1)) 
        cost = pd.Series(self.epochC2G, self.cost2goImprov)
        cost.to_csv('costsImprov.csv', header=None)



        
    def saveModel(self):
        print(50*"#","New better cost to go found! Saving Model", 50*"#")
        self.q.save_weights("DeepQNN.h5")
    
    def visualize(self, file_name, x=None):
        '''Visualize NN results loading model weights and letting it run for 10% of training epochs '''
        # Load NN weights from file
        self.q.load_weights(file_name)
        
        if x is None:
            x0 = x = self.env.reset()
        else:
            x0 = x
        
        costToGo = 0.0
        gamma = 1
        
        for i in range(int(h.EPOCHS/10)):
            pred = self.q.predict(x.reshape(1,-1))
            u = np.argmin(pred.reshape(self.nu,self.Joints), axis=0)
            if len(u) == 1:
                u = u[0]
            x, cost = self.env.step(u)
            costToGo += gamma*cost
            self.ctgRecord.append(costToGo)

            gamma *= h.GAMMA
            self.env.render()
        
        
        self.exportCosts(h.EPOCHS/10)
        
    

    def get_critic4(self):
        ''' Create the neural network with 4 hidden layers to represent the Q function '''
        inputs = layers.Input(shape=(self.nx))
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(self.Joints*self.nu)(state_out4)
    
        model = tf.keras.Model(inputs, outputs)
    
        return model

    def get_critic3(self):
        ''' Create the neural network with 3 hidden layers to represent the Q function '''
        inputs = layers.Input(shape=(self.nx))
        state_out1 = layers.Dense(64, activation="relu")(inputs) 
        state_out2 = layers.Dense(64, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        outputs = layers.Dense(self.Joints*self.nu)(state_out3)
    
        model = tf.keras.Model(inputs, outputs)
    
        return model

    def exportCosts(self, epochs):
        '''function used to export the costs for further data analysis in an external file'''
        cost = np.cumsum(self.ctgRecord)/range(1,int(epochs+1)) 
        cost = pd.Series(cost)
        cost.to_csv('costs.csv', header=None)


if __name__=='__main__':

    training = False
    file_name = "DeepQNN4LayersSingle.h5"  # Single pendulum model 
    #file_name = "DeepQNNDouble3.h5"        # Double pendulum model 

    deepQN = DeepQNN(11,4)
    if training:
        print(50*"#"); print("Beginning training ")
        deepQN.trainNN()

        if keyboard.is_pressed('s'): # manually stop training by pressing S so that the meaningful data of the model are safely saved 
            deepQN.saveModel()
            deepQN.exportCosts(deepQN.epochExport)
            deepQN.plotting(deepQN.epochExport)
            exit()

    else: 
        deepQN.visualize(file_name)
