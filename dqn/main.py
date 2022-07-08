import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from DQN import hyperParam as hp, DQ as DQ
import time 
import sys
import os 
import pandas as pd 

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np_config.enable_numpy_behavior()


def decay(minm, init, rate, count):
    
    new = minm + (init-minm) * np.exp(-rate*count)

    return new




@tf.function
def update(batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    
    # convert all to tensor objects
    xuBatch, costBatch, xuNBatch, rBatch = zip(*batch)
    xuNBatch = tf.convert_to_tensor(xuNBatch); xuBatch = tf.convert_to_tensor(xuBatch); costBatch = tf.convert_to_tensor(costBatch)

    with tf.GradientTape() as tape:
        targetValues = DqN.qTgt(xuNBatch, training=True)
        targetValuesInput = tf.squeeze(tf.math.reduce_sum(targetValues,axis=2))
        # Compute 1-step targets for the critic loss
        y = tf.TensorArray(tf.float64, size=Hp.batchSize, clear_after_read=False)
        for i, reached in enumerate(rBatch):
            if reached:
                # apply only cost of current step if at target state
                y = y.write(i,costBatch[i])
            else:
                y = y.write(i,costBatch[i] + Hp.disc*targetValuesInput[i])
        y= y.stack()             
        # Compute batch of Values associated to the sampled batch of states
        qValue = DqN.q(xuBatch, training=True) 
        qValueInput = tf.squeeze(tf.math.reduce_sum(qValue,axis=2))
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - qValueInput)) 

    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    deltaQ = tape.gradient(Q_loss, DqN.q.trainable_variables)
    # Update the critic backpropagating the gradients
    DqN.optimizer.apply_gradients(zip(deltaQ, DqN.q.trainable_variables))  
    return True
    

def uSelect():
    
    if np.random.uniform(0,1) < DqN.epsilon:
        u = tf.random.uniform(shape=[DqN.Joints],minval=0,maxval=Hp.nu,dtype=tf.dtypes.int64)
    else:
        xRep = tf.repeat(DqN.x.reshape(1,-1),Hp.nu**(DqN.Joints),axis=0)
        xuCk = tf.concat([xRep,DqN.U],axis=1).reshape(Hp.nu**(DqN.Joints),1,-1)
        pred = DqN.q.__call__(xuCk)
        uInd = tf.squeeze(tf.math.argmin(tf.math.reduce_sum(pred,axis=2), axis=0))
        u = DqN.U[uInd]
    return u


def targetCheck(x, cost):
    if cost <= DqN.thresC and (np.abs(x[DqN.nv:])<= DqN.thresV).all() :
        DqN.tgtCount +=1 
    else: 
       DqN.tgtCount = 0                  
    
    
    if DqN.tgtCount >= Hp.Up:
        DqN.reached = True 
        print("Target Reached! X:COST:U" ,x , ',', round(cost,6),',',u)
        DqN.decThreshold = True
    else:
        DqN.reached = False


def exportCosts(cost):
    cost = pd.Series(cost)
    cost.to_csv('costs.csv', header=None)

    


if __name__=='__main__':
     
    Hp = hp()                # Hyperparameters 
    DqN = DQ(4, True)        # initialize model, select pendulum and NN architecture   
    start= time.time()       # start time of simulation 
    plotting = True

    try:
        for epoch in range(1,Hp.epochs+1):
            DqN.reset()                              #reset enviroment and parameters
            for step in range(Hp.epochStep):

                u = uSelect()                    # choose the control based on epsilon value

                DqN.step(u)
                if (Hp.ctrl): targetCheck(DqN.x, DqN.cost)     # check and update if pendulum reached target state

                DqN.save4replay(u)                                       # save observation to replay buffer

                if DqN.steps % Hp.updateFreq == 0:
                    DqN.updateQTarget() 
                    
                # Sampling from replay buffer and train
                if len(DqN.repBuffer) >= Hp.bufferSize and DqN.step % Hp.samplintSteps == 0:
                    batch = np.random.sample(DqN.replay_buffer, Hp.batchSize)
                    update(batch)

                DqN.updateParams()
            

            if len(DqN.costTg) > Hp.print:
                avCTgo = np.average(DqN.costTg[-Hp.print:]) 
            else: 
               avCTgo= DqN.cTg

            if DqN.decThreshold:
                DqN.count_thresh +=1
                DqN.thresC = DqN.thresV = decay(Hp.thresMin , Hp.thresCost, Hp.thresDecay,DqN.count_thresh)

            # start finding the best ctg after 5% of the epochs has passed to ensure good exploration vs exploitation trade-off
            if avCTgo <= DqN.bCostTg and epoch > (0.05*Hp.epochs):
                print("Improved ctg found: New value", round(avCTgo,4), " previous best value: ", round(DqN.bCostTg,4))
                DqN.bCostTg = avCTgo

            # decay starts when the minimum size of the replay buffer has been reached
            if(len(DqN.repBuffer)>=Hp.minBuffer):
                DqN.countEpsilon +=1
                DqN.epsilon = decay(Hp.thresMin, Hp.epsilon,Hp.decEps,DqN.countEpsilon)
   
            DqN.costTg.append(DqN.cTg)
            
            if(epoch % Hp.print == 0):
                plt.plot( np.cumsum(DqN.costTg)/range(1,len(DqN.costTg)+1)  )
                plt.title ("Cost to go")
                plt.grid(True)
                plt.show()       
                
            if epoch % Hp.saveF == 0:
                DqN.saveModel(epoch)   
                
            if epoch % Hp.print == 0:
                dt = time.time() - start
                t = time.time()
                totT = t - start
                print('epoch: %d  cost: %.3f  buffer size: %d epsilon: %.4f threshold: %.5f elapsed time [s]: %.3f   total sim time [s]: %.1f ' % (epoch, avCTgo, len(DqN.repBuffer), DqN.epsilon,DqN.thresCost, dt, totT))
        


        cost = np.cumsum(DqN.costTg)/range(1,len(DqN.costTg)+1) 
        exportCosts(cost)

        if(plotting):
            plt.plot(cost)
            plt.scatter(cost)
            plt.xlabel("epoch Number")
            plt.ylabel("cost value")
            plt.title ("Average Cost to Go")
            plt.savefig("CostToGoTraining.eps")
            plt.grid(True)
            plt.show()
        
        print(25*"#")
        print('Training finished: saving model')
        DqN.q.saveModel(str(epoch))
            
    except KeyboardInterrupt:
        cost = np.cumsum(DqN.costTg)/range(1,len(DqN.costTg)+1) #TODO save costs for later plotting into a pandas data frame 
        exportCosts(cost)
        print(25*"#")
        print('Manually stopping training: saving model weights')
        DqN.q.saveModel(str(epoch))
            
        plt.plot( cost )
        plt.xlabel("epoch Number")
        plt.ylabel("cost value")
        plt.title ("Average Cost to Go")
        plt.grid(True)
        plt.savefig("CostToGoTraining.eps")
        
                
        
        
        
    
    
    
