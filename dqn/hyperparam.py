'''
In this file an environment with fixed and variable hyperparameters for the Deep Q Network is defined 
'''

class hyperParam:
    def __init__(self, nJoint=1, epochs=5000):
        #Fixed params for one joint
        self.nJoint = nJoint # Number of joints of the pendulum
        #Epochs define the number times that the agent goes to the entire dataset
        self.epochs = epochs #Number of cicles for training
        #batches to train in a single dataset to improve the accuracy of the model
        self.epochStep = 250  # The number of steps per epoch
        self.updateRate = self.epochs/5  # The frequncy for updating the Q-target weights
        #Buffers to store the trajectories in RL
        self.minBuffer = 5000  # The minimum size for the replay buffer to start training
        self.bufferSize = 50000 # The size of the replay buffer
        
        self.batchSize = 64     # The batch size taken randomly from buffer
        self.samplintSteps = 2   # The frequncy of sampling #DEBUG

        self.nu = 10   # number of discretized steps 
        self.Up = 100   # After how many steps the pendulum holds the standup position before considered being a success
        
        self.QLearn = 2e-3   # The learning rate of the DQN model 
        self.disc = 0.9  #  Discount factor - (GAMMA)
        self.epsilon = 1   # Initial probability of epsilon 
        self.decEps = 1e-3   # Exponential decrease rate for epsilon #DEBUG
        self.minEpsilon = self.decEps # The Minimum value for epsilon #DEBUG
        
        #limit parameters
        self.thresVel = 0.1         # Threshold forthe  velocity
        self.thresCost = 0.1       # Threshold for the cost
        self.thresMin = 1e-3          # Minimum value for threshold
        self.thresDecay = 0.003         # Decay rate for threshold in greedy q learning
        self.thresAngle = 0.1        # Threshold for the angle

        #Plotting params
        self.PLOT        = True          # Plot out results
        self.SAVE_MODEL  = True          # Save the models in files
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

    def up_nJoints(self, n):
        self.nJoint = n 
    
