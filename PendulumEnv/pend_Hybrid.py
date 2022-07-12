from pendulum import Pendulums #Simulation of the environment for a N-pendulum
import numpy as np



class HybridP:

    def __init__(self, Joints , nu=11, uMax=15, dt=0.1, ndt=1, noise_stddev=0):

        self.Joints  = Joints  # number of joints of the pendulum 
        self.Pend = Pendulums(self.Joints,0) # create a pendulum model -- 0 Gaussian noise assumed 
        self.Pend.DT  = dt  # time step lenght
        self.Pend.NDT = ndt # number of Euler steps per integration 
        self.nu = nu        # discretization steps for the joint torque control 
        self.nx = self.Pend.nx  # state dimension 

        if self.Joints== 2:  # max torque allowd -- 5% for double pendulum, 15% for single 
            self.uMax = uMax/3
        else:
            self.uMax= uMax


        self.DU = 2*uMax/nu # joints torque  discretization resolution 
    
    
    def CtoD(self, u):  # continous to discrete -- not used at all -- 
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        disc_u = np.floor((u+self.uMax)/self.DU).astype(int)
        return disc_u

    def DtoC(self, yi):  # function to switche the control from discreate to continous to allow the  state calculations
        #print("control", yi, type(yi))
        yi = np.clip(yi,0,self.nu-1) - (self.nu-1)/2
        u_cont = self.DU*yi
        return u_cont

    def reset(self, x=None):  # reset the state of the pendulum to a random velocity and position 
        self.x = self.Pend.reset(x)
        return self.x


    def step(self,yi): # calculates the next state and cost based on the chosen control
        #print("control", yi, type(yi))
        u = self.DtoC(yi)
        self.x, cost = self.Pend.step(u)
        return self.x, cost
    
    def render(self):
        self.Pend.render()

    
    
    
    
if __name__=="__main__":

    trial = HybridP()  
    print(trial.nu)
    print(50*"--")
    print(trial.nx)
    
