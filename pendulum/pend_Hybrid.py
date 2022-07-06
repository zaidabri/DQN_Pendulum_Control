from pendulum import Pendulum 
import numpy as np

class Single_Hpendulum:

    ''' We create a single pendulum environment. 
    In order to create an hybrid pendulum the Joint Angle, as well as the velocity are continous,
    while the torque applied to the joint is discretized with dis_steps 

    '''
    def __init__(self, nJoint=1, dis_steps=10, qMax=10, dt=0.1, ndt=1):

        self.N_pend = Pendulum(nJoint, 0)  # create a normal pendulum env first, we assume no noise in the model 
        self.dt = dt 
        self.N_pend.DT = self.dt
        self.ndt = ndt
        self.N_pend.NDT = self.ndt
        self.NJoint = nJoint
        self.steps = dis_steps  # number of discretization steps   # BUG  TODO FIX IT ASAP 
        self.NX = self.N_pend.nx 
        self.NV = self.N_pend.nv 
        self.qMax = qMax                    # % maximum torque in the system 
        self.DU = 2*self.qMax / self.steps  # we calculate the joint torque resolution 
    
    def reset(self):
        ''' resets state of the pendulum to a random position and velocity'''
        self.x = self.N_pend.reset(None)
        return self.x

    def render(self):
        self.N_pend.render()

    def ContinousCtrl(self, iu):
        ''' this function allows to 
        switch the control from discreate to continous and it is used for the 
        state calculations'''

        steps_ = self.steps -1 
        c = np.clip(iu, 0, steps_) 
        iu = c - steps_/2

        return iu*self.DU
    

    def step(self,iu):
        ''' We calculate the next state and cost'''
        u = self.ContinousCtrl(iu)

        if type(u) is np.ndarray: 
            u = u 
        else: 
           u = [u]
    
        self.x, cost = self.N_pend.step(u)
        return self.x, cost



class Double_Hpendulum:

    ''' We create a double pendulum environment. 
    In order to create an hybrid pendulum the Joint Angle, as well as the velocity are continous,
    while the torque applied to the joint is discretized with dis_steps 

    '''
    def __init__(self, nJoint=2, dis_steps=10, qMax=10, dt=0.1, ndt=1):

        self.N_pend = Pendulum(nJoint, 0)  # create a normal pendulum env first, we assume no noise in the model 
        self.dt = dt 
        self.N_pend.DT = self.dt
        self.ndt = ndt
        self.N_pend.NDT = self.ndt
        self.NJoint = nJoint
        self.steps = dis_steps  # number of discretization steps 
        self.NX = self.N_pend.nx 
        self.NV = self.N_pend.nv 
        self.qMax = qMax                    # % maximum torque in the system 
        self.DU = 2*self.qMax / self.steps  # we calculate the joint torque resolution 
    
    def reset(self):
        ''' resets state of the pendulum to a random position and velocity'''
        self.x = self.N_pend.reset(None)
        return self.x

    def render(self):
        self.N_pend.render()

    def ContinousCtrl(self, iu):
        ''' this function allows to 
        switch the control from discreate to continous and it is used for the 
        state calculations'''

        steps_ = self.steps -1 
        c = np.clip(iu, 0, steps_) 
        iu = c - steps_/2

        return iu*self.DU
    

    def step(self,iu):
        ''' We calculate the next state and cost'''
        u = self.ContinousCtrl(iu)

        if type(u) is np.ndarray: 
            u = u 
        else: 
           u = [u]
    
        self.x, cost = self.N_pend.step(u)
        return self.x, cost
