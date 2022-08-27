import sympy as sp
from sympy.abc import x,y,z
from scipy.constants import epsilon_0
import numpy as np

class PointCharge:
    def __init__(self, q, m, r0=[0,0,0], v0=[0,0,0]) -> None:
        self.q = q
        self.m = m
        self.r0 = r0
        self.v0 = v0


class ChargeDistribution:
    def __init__(self, num_values={}, *args, **kwargs) -> None:
        self._num_values = num_values
        self.num_Q = num_values.get('Q', 0)
        self._symbols = {
            'Q': sp.Symbol('Q', real=True),
            'eps0': sp.Symbol(r"\varepsilon_0", real=True, positive=True)
        }
        self.V = None
        self.num_V = None
        self.E = None
        self.num_E = None
    
    @property
    def Q(self):
        return self._symbols['Q']
    
    @property
    def eps0(self):
        return self._symbols['eps0']
    
    def initV(self):
        pass
    
    def initialize_V_and_E(self):
        self.initV()
        self.E = [
            -self.V.diff(c) for c in [x,y,z]
        ]
        self.num_V = np.vectorize(sp.lambdify((x,y,z), self.V.subs(self._subs), modules=['numpy', 'sympy']))
        self.num_E = [
            np.vectorize(
                sp.lambdify((x,y,z), Ei.subs(self._subs), modules=['numpy', 'sympy'])
            )
            for Ei in self.E
        ]
    
    def initSubs(self):
        self._subs = {
            self.Q: self.num_Q,
            self.eps0: epsilon_0
        }
    
    def addTraceToFigure(self, fig):
        pass


class UniformlyChargedRing(ChargeDistribution):
    def __init__(self, num_values={}, *args, **kwargs) -> None:
        super().__init__(num_values, *args, **kwargs)
        self._symbols['R'] = sp.Symbol('R', real=True, positive=True)
        self.num_R = self._num_values['R']
        self.num_center = self._num_values.get('center', (0,0,0))
        
        self.initSubs()
        self.initialize_V_and_E()
    
    def initSubs(self):
        super().initSubs()
        self._subs.update({
            self.R: self.num_R
        })
    
    @property
    def R(self):
        return self._symbols['R']

    def initV(self):
        x0, y0, z0 = self.num_center
        rho = sp.sqrt((x0+x)**2 + (y0+y)**2 + (z0+z)**2)
        self.V = self.Q/(2 * sp.pi**2 * self.eps0 * sp.sqrt((rho + self.R)**2 + z**2)) * \
                 sp.elliptic_k(4*rho*self.R/((rho+self.R)**2 + z**2))

    def addTraceToFigure(self, fig):
        pass

    
    
    