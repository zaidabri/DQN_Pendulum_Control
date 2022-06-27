import numpy as np
from pendulum import Pendulum 

class Hybrid_Pendulum:
    ''' Initializes a Pendulum environment. The state, which icludes Joint angle, 
        and velocity are both continuous. Control (joint torque) gets discretized 
        to the specified steps
    '''
    def __init__(self, nbJoint=1, nu=11, uMax=5, dt=0.2, ndt=1, noise_stddev=0):
        self.nbJoint  = nbJoint
        self.pendulum = Pendulum(self.nbJoint,noise_stddev)
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.nu = nu                 # Number of discretization steps for control
        self.nx = self.pendulum.nx
        self.nv = self.pendulum.nv
        self.uMax = uMax             # Max torque allowed
        self.DU = 2*uMax/nu          # resolution for joint torque
    
    def d2cu(self, iu):
        ''' switches control from discreate to continous for state calculations'''
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU
    
    def reset(self, x=None):
        ''' resets state to a random position and velocity'''
        self.x = self.pendulum.reset(x)
        return self.x

    def step(self,iu):
        ''' Calculates the next state and cost based on the chosen control '''
        u = self.d2cu(iu)
        u = u if type(u) is np.ndarray else [u]
        self.x, cost = self.pendulum.step(u)
        return self.x, cost
    
    def render(self,slow_down=False):
        self.pendulum.render(slow_down=slow_down)
        

if __name__ == "__main__":
    print("starting test..")
    env = Hybrid_Pendulum()
    nu = env.nu 
    x = env.reset()
    print('x:\n', x)

