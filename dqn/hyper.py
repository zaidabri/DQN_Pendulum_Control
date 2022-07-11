


SAMPLING_STEPS         = 4             # Steps to sample from replay buffer
BATCH_SIZE             = 64            # Batch size sampled from replay buffer
REPLAY_BUFFER_SIZE     = 50000         # Size of replay buffer
MIN_BUFFER_SIZE        = 5000          # Minimum buffer size to start training
UPDATE_Q_TARGET_STEPS  = 50          # Steps to update Q target  # previously 60 | 200 | 
EPOCHS              = 5000            # Number of training EPOCHS # previously 5000
MAX_EPOCH_LENGTH     = 100            # Max EPOCH length
QVALUE_LEARNING_RATE   = 0.05        # Learning rate of DQN # previously 0.05 | 0.9
GAMMA                  = 0.9           # Discount factor 
EPSILON                = 1             # Initial exploration probability of eps-greedy policy
EPSILON_DECAY          = 0.01         # Exploration decay in exponential decreasing
MIN_EPSILON            = EPSILON_DECAY       # Minimum of exploration probability