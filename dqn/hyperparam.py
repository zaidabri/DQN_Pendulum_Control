'''
In this file an environment with fixed and variable hyperparameters for the Deep Q Network is defined 
'''

class hyperParam:
    def __init__(self, nJoint=1, epochs=5000):
        
        self.nJoint = nJoint # Number of joints of the pendulum
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
        self.thres_vel = 0.1         # Threshold forthe  velocity
        self.thres_C   = 0.1       # Threshold for the cost
        self.Min_Trsh  = 1e-3          # Minimum value for threshold
        self.Trsh_decay = 0.003         # Decay rate for threshold
        self.PLOT        = True          # Plot out results
        self.SAVE_MODEL  = True          # Save the models in files
        self.ctrl  = False         # Check if target state is reached
    
    def up_saveF(self, n):
        self.saveF = n 

    def up_epochLen(self, n):
        self.epochLen = n 

    def up_samplsteps(self, n):
        self.sampling_stps = n 

    def up_target_freq(self, n):
        self.updt_target = n 

    def up_nJoints(self, n):
        self.nJoint = n 

