import numpy as np
from scipy import sparse
import ctypes



class Diffusion:
    def __init__(self,grid,par=None,bc = 3) -> None:
        self.grid = grid
        par = np.hstack([[1]*9,[1]*9,[1]*9])
        self.bc = bc
        
    def Ah(self,H = None, bc = 3) -> sparse.csc_matrix:
        if bc == 1:
            lib = ctypes.cdll.LoadLibrary('./libAHb1.so')
        elif bc == 2:
            lib = ctypes.cdll.LoadLibrary('./libAHb2.so')
        else:
            lib = ctypes.cdll.LoadLibrary('./libAHb3.so')
        fnew = lib.AH_new
        fnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=4,shape = (M*N,4,2,2)), ctypes.c_double,ctypes.c_double]
        fnew.restype = ctypes.c_void_p
        obj = fnew(self.grid.M, self.grid.N, H, self.grid.hx, self.grid.hy)


        frow = lib.AH_Row
        frow.argtypes = [ctypes.c_void_p]
        frow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.grid.M*self.grid.N*9,))
        row = frow(obj)

        fcol = lib.AH_Col
        fcol.argtypes = [ctypes.c_void_p]
        fcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.grid.M*self.grid.N*9,))
        col = fcol(obj)

        fval = lib.AH_Val
        fval.argtypes = [ctypes.c_void_p]
        fval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (self.grid.M*self.grid.N*9,))
        val = fval(obj)

        rem = row != (self.grid.M*self.grid.N)
        row = row[rem]
        col = col[rem]
        val = val[rem]
        res = sparse.csc_matrix((val, (row, col)), shape=(self.grid.M*self.grid.N, self.grid.M*self.grid.N))

        fdel = lib.AH_delete
        fdel.argtypes = [ctypes.c_void_p]
        fdel.restype = None
        fdel(obj)
        return(res)
    
    def dAh(self) -> sparse.csc_matrix:
        return
    
    def H(self,par = None) -> np.ndarray:
        pg = np.exp(self.grid.evalBH(par = gamma))
        pvx = self.grid.evalBH(par = vx)
        pvy = self.grid.evalBH(par = vy)
        aV = np.sqrt(pvx**2 + pvy**2)
        mV = np.array([[pvx,pvy],[pvy,-pvx]]).T.swapaxes(0,1)
        cosh_aV = (np.exp(aV)+ np.exp(-aV))/2
        sinh_aV = (np.exp(aV)- np.exp(-aV))/2
        H = (pg*cosh_aV)[:,:,np.newaxis,np.newaxis]*np.eye(2) + (pg*sinh_aV/aV)[:,:,np.newaxis,np.newaxis]*mV
        return(H)
    
    def dH(self,gamma=None,vx = None,vy = None,d=None,grad = False):
        dpar = np.zeros(9)
        dpar[d] = 1

        pg = np.exp(self.grid.evalBH(par = gamma))
        pvx = self.grid.evalBH(par = vx)
        pvy = self.grid.evalBH(par = vy)
        aV = np.sqrt(pvx**2 + pvy**2)
        mV = np.array([[pvx,pvy],[pvy,-pvx]]).T.swapaxes(0,1)
        cosh_aV = (np.exp(aV)+ np.exp(-aV))/2
        sinh_aV = (np.exp(aV)- np.exp(-aV))/2
        
        dpg = self.grid.bsH[:,:,d]*pg
        H_gamma = (dpg*cosh_aV)[:,:,np.newaxis,np.newaxis]*np.eye(2) + (dpg*sinh_aV/aV)[:,:,np.newaxis,np.newaxis]*mV

        dpvx = self.grid.evalBH(par = dpar)
        dmV = np.array([[dpvx,pvy*0],[pvy*0,-dpvx]]).T.swapaxes(0,1)
        H_vx = (pg*sinh_aV*dpvx*pvx/aV)[:,:,np.newaxis,np.newaxis]*np.eye(2) + (pg*(cosh_aV*pvx*dpvx - sinh_aV*pvx*dpvx/aV)/aV**2)[:,:,np.newaxis,np.newaxis]*mV + (pg*sinh_aV/aV)[:,:,np.newaxis,np.newaxis]*dmV
        

        dpvy = self.grid.evalBH(par = dpar)
        dmV = np.array([[pvx*0,dpvy],[dpvy,-pvx*0]]).T.swapaxes(0,1)
        H_vy = (pg*sinh_aV*dpvy*pvy/aV)[:,:,np.newaxis,np.newaxis]*np.eye(2) + (pg*(cosh_aV*pvy*dpvy - sinh_aV*pvy*dpvy/aV)/aV**2)[:,:,np.newaxis,np.newaxis]*mV + (pg*sinh_aV/aV)[:,:,np.newaxis,np.newaxis]*dmV
        return((H_gamma,H_vx,H_vy))

    def print(self,par):
        return("| \u03BA = %2.2f"%(np.exp(par[0:9]).mean()) +  ", \u03B3 = %2.2f"%(np.exp(par[9:18]).mean()) + ", vx = %2.2f"%(par[18:27].mean()) + ", vy = %2.2f"%(par[27:36].mean()) +  ",\u03C4 = %2.2f"%(np.exp(par[36])))
    
    
    def plot(self):
        return
