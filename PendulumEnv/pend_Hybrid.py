from pendulum import Pendulums #Simulation of the environment for a N-pendulum
import numpy as np



class HybridP:

    def __init__(self, Joints =2 , nu=11, uMax=5, dt=0.1, ndt=1, noise_stddev=0):

        self.Joints  = Joints
        self.Pend = Pendulums(self.Joints,0)
        self.Pend.DT  = dt
        self.Pend.NDT = ndt
        self.nu = nu        
        self.nx = self.Pend.nx
        self.uMax = uMax   
        self.DU = 2*uMax/nu 
    
    def render(self):
        self.Pend.render()

    
    def reset(self, x=None):
        self.x = self.Pend.reset(x)
        return self.x


    def DtoC(self, yi):
        #print("control", yi, type(yi))
        yi = np.clip(yi,0,self.nu-1) - (self.nu-1)/2
        u_cont = self.DU*yi
        return u_cont


    def step(self,yi):
        #print("control", yi, type(yi))
        u = self.DtoC(yi)
        self.x, cost = self.Pend.step(u)
        return self.x, cost
    

    
    def CtoD(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        disc_u = np.floor((u+self.uMax)/self.DU).astype(int)
        return disc_u
    
if __name__=="__main__":

    trial = HybridP()  
    print(trial.nu)
    print(50*"--")
    print(trial.nx)
    
