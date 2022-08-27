import sympy as sp
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from physics_applets.electrostatics.chargedistr import ChargeDistribution, PointCharge

class ElectrostaticSystem:
    def __init__(self, qdist: ChargeDistribution, qpoint: PointCharge) -> None:
        self.qdist = qdist
        self.qpoint = qpoint
    
    def equationOfMotion(self, u, t):
        x,y,z,vx,vy,vz = u
        a = [
            float(self.qpoint.q*Ei(x,y,z)/self.qpoint.m)
            for Ei in self.qdist.num_E
        ]
        return [vx,vy,vz,*a]
    
    def evolveInTime(self, t_arr):
        u0 = [*self.qpoint.r0, *self.qpoint.v0]
        res = odeint(self.equationOfMotion, u0, t_arr)
        df_res = pd.DataFrame(res, columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
        df_res.insert(0, 't', t_arr)
        return df_res
        