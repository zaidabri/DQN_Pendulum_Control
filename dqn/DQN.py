import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PendulumEnv import pend_Hybrid #Debugged
import collections
import matplotlib.pyplot as plt 
import time 
import random
import pandas as pd 

import collections
from tensorflow.python.ops.numpy_ops import np_config



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np_config.enable_numpy_behavior()

class hyperParam:
    def __init__(self, epochs=5000):
        
        #Epochs define the number times that the agent goes to the entire dataset
        self.epochs = epochs #Number of cicles for training
        #batches to train in a single dataset to improve the accuracy of the model
        self.epochStep = 250  # The number of steps per epoch
        self.uptRate = 5 # The frequncy for updating the Q-target weights
        self.updateFreq = self.epochs/self.uptRate  # The frequncy for updating the Q-target weights
        self.batchSize = 64     # The batch size taken randomly from buffer
        self.samplintSteps = 2   # The frequncy of sampling #DEBUG

        #Buffers to store the trajectories in RL
        self.minBuffer = 5000  # The minimum size for the replay buffer to start training
        self.bufferSize = 50000 # The size of the replay buffer
        
        self.nu = 10   # number of discretized steps 
        self.Up = 100   # After how many steps the pendulum holds the standup position before considered being a success
        
        self.QLearn = 2e-3   # The learning rate of the DQN model 
        self.gamma = 0.9  #  Discount factor - (GAMMA)
        self.epsilon = 1   # Initial probability of epsilon 
        self.decEps = 1e-3   # Exponential decrease rate for epsilon #DEBUG
        self.minEpsilon = self.decEps # The Minimum value for epsilon #DEBUG
        
        #limit parameters
        self.thresVel = 0.1         # Threshold forthe  velocity
        self.thresCost = 0.1       # Threshold for the cost
        self.thresMin = 1e-3          # Minimum value for threshold
        self.thresDecay = 0.003         # Decay rate for threshold in greedy q learning
        self.thresAngle = 0.1        # Threshold for the angle

        self.ctrl  = False         # Check if target state is reached
        self.saveF = 200 # The number of steps needed before saving a model
        self.print = 20  # print out freq

    def up_saveF(self, n):
        self.saveF = n 

    def eps_decay(self, n):
        # update epsilon decay in epsilon-greedy policy 
        self.decEps = n 

    def up_epochStep(self, n):
        self.epochStep = n 

    def up_samplsteps(self, n):
        self.samplintSteps = n 

    def up_target_freq(self, n):
        self.updateRate = n 

    
    




class DQ():

    def __init__(self, NN_layers):
        # initialize enviroment
        self.Hpar = hyperParam()
        
        self.enviro = pend_Hybrid.Double_Hpendulum()
        self.Joints = 2 

        self.nx = self.enviro.NX
        self.nv = self.enviro.NV     
        self.nu = self.Hpar.nu
        

        if NN_layers == 4:
            self.q = self.get_critic4()
            self.NN_layers = NN_layers
            self.q.summary()
            self.qT = self.get_crtitic4()
            self.qT.set_weights(self.q.get_weights())

        elif NN_layers == 6:

            self.q = self.get_critic6()
            self.NN_layers = NN_layers
            self.q.summary()
            self.qT = self.get_crtitic6()
            self.qT.set_weights(self.q.get_weights())

        else: 
            self.q = self.get_critic3()
            self.q.summary()
            self.NN_layers = 3
            self.qT = self.get_crtitic3()
            self.qT.set_weights(self.q.get_weights())


        
        self.repBuffer = collections.deque(maxlen = self.Hpar.bufferSize)
        self.epsilon = self.Hpar.epsilon

        self.cTg = 0                     
        self.bCostTg = np.inf  
        self.costTg = []  
        self.steps = 0  
        self.optimizer = tf.keras.optimizers.Adam(self.Hpar.QLearn)
        
    
    def chooseU(self, x, eps):

        if np.random.uniform(0,1) < eps: 
            u = np.randint(self.Hpar.nu, size=self.Joints)
        else: 
            prediction = self.q.predict(x.reshape(1,-1))
            u = np.argmin(prediction.reshape(self.Hpar.nu, self.Joints), axis = 0)



    def trainNN(self): 
        Start = time.time()

        steps = 0
        eps = self.epsilon
        thres = self.Hpar.epochStep -1 

        for epoch in range(self.Hpar.epochs): 
            
            ctgo = 0; x= self.enviro.reset(); gamma =1 

            for step in range(self.Hpar.epochStep): 
                step = step+1

                u = self.chooseU(x, eps)
                xNext, cost = self.enviro.step(u)

                if step == thres: 
                    finished = True 
                else: 
                    finished = False 

                self.repBuffer.append([x, u, cost, xNext, finished])

                if steps % self.Hpar.updateFreq == 0: 
                    self.qT.set_weights(self.q.get_weights())

               
                if len(self.repBuffer) > self.Hpar.minBuffer and steps % self.Hpar.samplintSteps == 0:
                    batch = random.sample(self.repBuffer, self.Hpar.batchSize)
                    self.update(batch)
                
                x = xNext
                ctgo += gamma * cost
                gamma *= self.Hpar.gamma
                
            
            if ctgo < self.bCostTg:

                self.bCostTg = ctgo
                self.Q.save_weights("model"+str(self.NN_layers)+"dqn.h5")
                
            
            eps= max(self.Hpar.minEpsilon, np.exp(-self.Hpar.decEps*epoch))
            self.costTg.append(ctgo)
            if epoch % self.Hpar.print == 0:
               
                print('Epoch: %d | cost %.5f | %.4f Epsilon | time elapsed %.3f [s]' % (epoch, np.mean(self.h_ctg[-self.Hpar.print:]), eps, (time.time-Start)))
                Start = time.time()

        print("--------------- Plotting & Exporting costs overview --------------------------")
        self.exportCosts()
        self.plotting()

    def batchSMPL(self, n, batch):
        arr = np.array([random.sample[n]] for random.sample in batch)
        return arr 


    def update(self, batch): 
        xBatch = self.batchSMPL(0, batch)
        uBatch = self.batchSMPL(1, batch)
        costBatch = self.batchSMPL(2, batch)
        xNextBatch = self.batchSMPL(3, batch)
        finishedBatch = self.batchSMPL(4, batch)

        with tf.GradientTape as tape: 

            TarOut = self.qT(xNextBatch, training = True).reshape((len(batch), -1, self.Joints))
            
            TarVal  = tf.math.reduce_sum(np.min(TarOut, axis=1), axis=1)
            
           
            y = np.zeros(len(batch))
            for id, finished in enumerate(finishedBatch):
                if finished != False:
                    y[id] = costBatch[id]
                else:
                    fctr = self.Hpar.gamma*TarVal[id]
                    y[id] = costBatch[id] + fctr      
            
           
            qOut = self.q(xBatch, training=True).reshape((len(batch),-1,self.Joints))

            a = np.repeat(np.arange(len(batch)),self.Joints).reshape(len(batch),-1)
            b = uBatch.reshape(n,-1)
            c = np.repeat(np.arange(self.Joints).reshape(1,-1),len(batch),axis=0)

            qVal  = tf.math.reduce_sum(qOut[a, b, c], axis=1)
            qLoss = tf.math.reduce_mean(tf.math.square(y - qVal))
        
        trainable = self.q.trainable_variables
        DeltaQ = tape.gradient(qLoss, trainable)
        self.optimizer.apply_gradients(zip(DeltaQ, trainable))
        


    def exportCosts(self):
        cost = np.cumsum(self.q.costTg)/range(1,self.Hpar.epochs+1) 
        cost = pd.Series(cost)
        cost.to_csv('costs.csv', header=None)

    def plotting(self):
        cost = np.cumsum(self.q.costTg)/range(1,self.Hpar.epochs+1) 
        plt.plot(cost, color = 'black')
        plt.title("average cost-to-go per episode")
        plt.xlabel("epoch number")
        plt.ylabel("cost to go value")
        plt.legend("cost-to-go")
        plt.grid(True)
        plt.savefig("costToGo.eps")
        plt.show()

    def visualize(self, file_name, x=None): 

        self.q.load_weights(file_name)

        gamma = 1 
        ctgo = np.float64(0)

        if x != None: 
            x = self.enviro.reset()
         
        x0 = x 

        for epoch in range(300): 
            prediction = self.q.predict(x.reshape(1,-1))
            u = np.argmin(prediction.reshape(self.Hpar.nu, self.Joints), axis = 0)
            k = len(u)
            if k == 1: 
                u = u[0]
            
            x, cost = self.enviro.step(u)
            ctgo = ctgo + (gamma*cost)
            gamma = gamma*self.Hpar.gamma

            self.enviro.render()

        print(70*"#"); print("state :", x, "| cost to go :", ctgo); print(70*"#")


    def get_critic3(self):
        
        inputs = layers.Input(shape=(1,self.nx+self.Joints),batch_size=self.Hpar.batchSize)
        state_out1 = layers.Dense(64, activation="relu")(inputs) 
        state_out2 = layers.Dense(64, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        outputs = layers.Dense(self.Joints)(state_out3)
    
        model = tf.keras.Model(inputs, outputs)
    
        return model

    def get_critic4(self):
        
        inputs = layers.Input(shape=(1,self.nx+self.Joints),batch_size=self.Hpar.batchSize)
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(self.Joints)(state_out4)
    
        model = tf.keras.Model(inputs, outputs)
    
        return model

    def get_critic6(self): 
        inputs = layers.Input(shape=(1,self.nx+self.Joints),batch_size=self.Hpar.batchSize)
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(32, activation="relu")(state_out2) 
        state_out4 = layers.Dense(32, activation="relu")(state_out3)
        state_out5 = layers.Dense(32, activation="relu")(state_out4)
        state_out6 = layers.Dense(64, activation="relu")(state_out5)
        outputs = layers.Dense(1)(state_out6) 

        model = tf.keras.Model(inputs, outputs)
    
        return model

    
if __name__=="__main__":

    deepQN = DQ(4)
    training = True 

    if deepQN.single:
       print("Deep Q network for single pendulum with hiden layers",deepQN.NN_layers )
    else:
       print("Deep Q network for double pendulum", deepQN.NN_layers )
   
    print(50*"--")
    if training == True:
        print("Begin training DQN ")
        deepQN.trainNN()

    #file_name = ""
    #deepQN.visualize()