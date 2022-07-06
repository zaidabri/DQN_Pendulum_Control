import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint
from pendulum import pend_Hybrid


import collections
from tensorflow.python.ops.numpy_ops import np_config
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np_config.enable_numpy_behavior()

class hyperParam:
    def __init__(self, epochs=5000):
        
       
        self.epochs = epochs
        self.epochLen = 250  # The number of steps per epoch
        self.buff_size = 50000 # The size of the replay buffer
        self.Min_buffer = 5000  # The minimum size for the replay buffer to start training
        self.batch_size = 64     # The batch size taken randomly from buffer
        self.sampling_stps = 2   # The frequncy of sampling
        self.updt_target = self.epochs/5  # The frequncy for updating the Q-target weights
        self.Q_learn = 2e-3   # The learning rate of the DQN model 
        self.Disc = 0.9  #  Discount factor - (GAMMA)
        self.Epslison = 1   # Initial probability of epsilon 
        self.Eps_decay = 1e-3   # The exponintial decay rate for epsilon
        self.Min_epsilon = self.Eps_decay # The Minimum value for epsilon
        self.saveF = 200 # The number of steps needed before saving a model
        self.print = 20  # print out freq
        self.nu = 10   # number of discretized steps 
        self.Up = 100   # After how many steps the pendulum holds the standup position before considered being a success
        self.thres_An = 0.1        # Threshold for the angle
        self.thres_vel = 0.15         # Threshold for the  velocity
        self.thres_C   = 0.1       # Threshold for the cost
        self.Min_Trsh  = 1e-3          # Minimum value for threshold
        self.Trsh_decay = 0.003         # Decay rate for threshold
        self.ctrl  = False         # Check if target state is reached
    
    def up_saveF(self, n):
        self.saveF = n 

    def eps_decay(self, n):
        # update epsilon decay in epsilon-greedy policy 
        self.Eps_decay = n 

    def up_epochLen(self, n):
        self.epochLen = n 

    def up_samplsteps(self, n):
        self.sampling_stps = n 

    def up_target_freq(self, n):
        self.updt_target = n 

    




class DQ():
    def __init__(self, NN_layers, single):
        # initialize enviroment
        self.Hpar = hyperParam()

        

        if single:
            self.enviro = pend_Hybrid.Single_Hpendulum()
            self.single = True
            self.Joints = 1 
        else: 
            self.enviro = pend_Hybrid.Double_Hpendulum()
            self.single = False 
            self.Joints = 2 

        self.nx = self.enviro.NX
        self.nv = self.enviro.NV     
        # initialize keras neural networks and optimizer

        if NN_layers == 4:
            self.q = self.get_critic4(False)
       
            self.qTgt = self.get_critic4(True)
            self.qTgt.set_weights(self.q.get_weights())
            self.NN_layers = NN_layers

        elif NN_layers == 6:

            self.q = self.get_critic6(False)
       
            self.qTgt = self.get_critic6(True)
            self.qTgt.set_weights(self.q.get_weights())
            self.NN_layers = NN_layers

        else: 
            print("default implementation")
            self.q = self.get_critic4(False)
            self.qTgt = self.get_critic4(True)
            self.qTgt.set_weights(self.q.get_weights())
            self.NN_layers = 4
        
        self.optimizer = tf.keras.optimizers.Adam(self.Hpar.qLearn)

        # initialize the replay buffer using deque
        self.repBuffer = collections.deque(maxlen = self.Hpar.buffSize)
        # initialize hyper paramters booleans and counters to calculate decay
        self.epsilon = self.Hpar.Epsilon
        self.thresC = self.Hpar.thresC
        self.thresV = self.Hpar.thresVel
        self.countEpsilon = 0
        self.counThresh = 0
        self.decThreshold = False              # a flag to signal that the thresholds need to be decreased
        
        self.GoalR = False                    # a boolean to check that pendulum reached target state
        self.tgtCount = 0                      # a count of the number of steps the pendulum stayed at target
        
        self.U = self.create_u_matrix()    # creating a matrix for descritized controls based on JOINT_COUNT
        self.costTg = []                         # keep a list of all cost to go per episode
        self.bCostTg = np.inf   
        self.steps = 0                    # keep track of all the steps taken across the episodes

        
        
    def reset(self):
        def param_rst(): # hyperParameters reset 
            self.decThreshold = False          # a flag to signal that the thresholds need to be decreased
            self.tgtCount = 0                  # a count of the number of steps the pendulum stayed at target
            self.costTg = np.float64(0)
            self.gammaI = 1

        self.x = self.enviro.reset()
        param_rst()
        
    
    def step(self,u): 
        self.x_next, self.cost = self.enviro.step(u)
    
    def update_q_target(self): 
        self.qTgt.set_weights(self.q.get_weights())

    
    def updateGamma(self):
       self.gammaI = self.gammaI * self.Hpar.Disc

    def update_params(self):
        self.steps += 1
        self.x = self.x_next
        self.costTg = self.costTg + (self.gammaI * self.cost)
        self.updateGamma()
        
        
   

    def save_model(self,epoch):
        if self.single:
            name = "ModelSingle@Epochs" + str(epoch)+ ".h5"
        else:
            name = "ModelDouble@Epochs" + str(epoch)+ ".h5"

        self.q.save_weights(name)
        
    
    def save_to_replay(self, u):
        xu = np.c_[self.x.reshape(1,-1),u.reshape(1,-1)]
        xu_next = np.c_[self.x_next.reshape(1,-1),u.reshape(1,-1)]
        to_append = [xu, self.cost, xu_next,self.reached]
        self.replay_buffer.append(to_append)


    def createUTable(self):
        

        u_listE= np.array(range(0, int(self.Hpar.nu)))
        fctr = np.power(self.Hpar.nu, (self.Joints - 1))
        for ctrl in range(self.Joints-1):
            if ctrl== self.Joints-2:
                u_listF = np.tile(np.array(range(0, int(self.Hpar.nu))), fctr)            
            else:
                u_listF = np.repeat(u_listE, np.power(self.Hpar.nu, (self.Joints-2-ctrl)))
                u_listF = np.tile(u_listF, np.power(self.Hpar.nu, (ctrl+1)))

        uTable = np.repeat(u_listE, np.power(self.Hpar.nu, (self.Joints-1)))

        uTable = np.c_[uTable,u_listF]

        return uTable

    def get_critic4(self, Q_tgt):
        
        inputs = layers.Input(shape=(1,self.nx+self.Joints),batch_size=self.Hpar.batch_size)
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(self.Joints)(state_out4)
    
        if Q_tgt:
            model = tf.keras.Model(inputs, outputs, name="QTargets")
        else: 
            model = tf.keras.Model(inputs, outputs, name="QFunc")
    
        return model

    def get_critic6(self, Q_tgt): 
        inputs = layers.Input(shape=(1,self.nx+self.Joints),batch_size=self.Hpar.batch_size)
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(32, activation="relu")(state_out2) 
        state_out4 = layers.Dense(32, activation="relu")(state_out3)
        state_out5 = layers.Dense(32, activation="relu")(state_out4)
        state_out6 = layers.Dense(64, activation="relu")(state_out5)
        outputs = layers.Dense(1)(state_out6) 

        if Q_tgt:
            model = tf.keras.Model(inputs, outputs, name="QTargets")
        else: 
            model = tf.keras.Model(inputs, outputs, name="QFunc")

        return model

    
if __name__=="__main__":
    deepQN = DQ(4,True)

    print(25*"#")
    if deepQN.single:
        print("Deep Q network for single pendulum with" + deepQN.NN_layers + "hidden layers")
    else:
        print("Deep Q network for double pendulum" + deepQN.NN_layers + "hidden layers")

    print(25*"#")
