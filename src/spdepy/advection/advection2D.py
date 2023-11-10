import numpy as np
from scipy import sparse
import ctypes
import os

class Advection2D:
    def __init__(self,grid, par=None, bc = 3) -> None:
        self.grid = grid
        par = np.array([1,1]) if par is None else par
        self.setPars(par)
        self.bc = bc
        self.setClib()
        
    def __call__(self, par):
        ws = np.array([par[0],par[1]])
        Aw = self.Aw(ws)
        return(Aw)
    
    def D(self,par,full = False):
        dAw = []
        for i in range(2):
            ws = np.zeros(2)
            ws[i] = 1
            dAw.append(self.Aw(ws))
        if full:
            Aw = self.Aw(np.array([par[0],par[1]]))
            return(Aw,dAw)
        else:
            return(dAw)
    
    def Aw(self,ws):
        obj = self.Awnew(self.grid.M, self.grid.N, ws, self.grid.hx, self.grid.hy)
        row = self.Awrow(obj)
        col = self.Awcol(obj)
        val = self.Awval(obj)
        self.Awdel(obj)
        rem = row != (self.grid.M*self.grid.N)
        row = row[rem]
        col = col[rem]
        val = val[rem]
        return(sparse.csc_matrix((val, (row, col)), shape=(self.grid.M*self.grid.N, self.grid.M*self.grid.N)))
        
    def getPars(self):
        return(np.hstack([self.wx, self.wy]))
    
    def setClib(self) -> None:
        if not os.path.exists('./ccode/lib_Acw_2D_b%d.so'%(self.bc)):
            os.system('g++ -c -fPIC ./ccode/Acw_2D_b%d.cpp -o ./ccode/Acw_2D_b%d.o'%(self.bc,self.bc))
            os.system('g++ -shared -o ./ccode/lib_Acw_2D_b%d.so ./ccode/Acw_2D_b%d.o'%(self.bc,self.bc))
            os.system('rm ./ccode/Acw_2D_b%d.o'%(self.bc))
        self.lib = ctypes.cdll.LoadLibrary('./ccode/lib_Acw_2D_b%d.so'%(self.bc))

        self.Awnew = self.lib.Aw_new
        self.Awnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,shape = (2,)), ctypes.c_double,ctypes.c_double]
        self.Awnew.restype = ctypes.c_void_p
        self.Awrow = self.lib.Aw_Row
        self.Awrow.argtypes = [ctypes.c_void_p]
        self.Awrow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.grid.M*self.grid.N*5,))
        self.Awcol = self.lib.Aw_Col
        self.Awcol.argtypes = [ctypes.c_void_p]
        self.Awcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.grid.M*self.grid.N*5,))
        self.Awval = self.lib.Aw_Val
        self.Awval.argtypes = [ctypes.c_void_p]
        self.Awval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (self.grid.M*self.grid.N*5,))
        self.Awdel = self.lib.Aw_delete
        self.Awdel.argtypes = [ctypes.c_void_p]
        self.Awdel.restype = None
        
    def setPars(self,par)-> None:
        self.wx = par[0]
        self.wy = par[1]
        
    def print(self,par):
        return("wx = %2.2f"%(par[0]) + ", wy = %2.2f"%(par[1]))

    def plot(self):
        pass
    