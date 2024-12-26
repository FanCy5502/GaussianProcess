from abc import ABC, abstractmethod
import numpy as np
from utils import dist_spuare


class Kernel(ABC):
    # 基础径向基函数
    def __init__(self, l, s):
        self.l = l
        self.s = s

    @abstractmethod
    def __call__(self, x1, x2):
        pass


class RBF(Kernel):
    def __call__(self, x1, x2):
        return self.l*np.exp(-self.s*dist_spuare(x1,x2))


class Matern(Kernel):
    # Matern核函数, 相较于基础的rbf核函数, 增加了一个参数v, 更加平滑
    def __init__(self, l, s, v):
        super().__init__(l, s)
        self.v = v

    def __call__(self, x1, x2):
        d = np.sqrt(dist_spuare(x1,x2))
        if self.v == 0.5:
            return self.l*np.exp(-self.s*d)
        elif self.v == 1.5:
            return self.l*(1+self.s*d)*np.exp(-self.s*d)
        elif self.v == 2.5:
            return self.l*(1+self.s*d+self.s**2*d**2/3)*np.exp(-self.s*d)
        else:
            raise ValueError('v must be 0.5, 1.5 or 2.5')


class ExpSineSquared(Kernel):
    def __init__(self, l, p, s):
        super().__init__(l, s)
        self.p = p

    def __call__(self, x1, x2):
        return self.l*np.exp(-2*self.s*np.sin(np.pi*dist_spuare(x1,x2)/self.p)**2)


class RationalQuadratic(Kernel):
    def __init__(self, l, a, s):
        super().__init__(l, s)
        self.a = a

    def __call__(self, x1, x2):
        return self.l*(1+dist_spuare(x1,x2)/(2*self.a))**(-self.a)


class WhiteKernel(Kernel):
    def __init__(self, l):
        super().__init__(l, 0)

    def __call__(self, x1, x2):
        return self.l if x1 == x2 else 0


if __name__ == '__main__':
    x = np.linspace(-np.pi, np.pi, 100)
    rbf(x,x,*(1,2))
