import numpy as np
from scipy import sparse
import ctypes
import os


class Anisotropic2D:
    def __init__(self,grid = None, par=None, bc = 3) -> None:
        assert grid is not None, "Grid is not defined"
        self.grid = grid
        self.type = "Constant Anisotropic Diffusion in 2D"
        par = np.array([-1,1,1,1]) if par is None else par
        self.bc = bc
        self.setPars(par)
        self.setClib()
        
    def __call__(self, par):
        Dk = np.exp(par[0])*sparse.eye(self.grid.Ns)
        H = self.H(par)
        AH = self.Ah(H)
        Dv = self.grid.V*sparse.eye(self.grid.Ns)
        return(Dk@Dv - AH)
        
        
    def D(self,par,full = False):
        Dk = np.exp(par[0])*sparse.eye(self.grid.Ns)
        Dv = self.grid.V*sparse.eye(self.grid.Ns)
        if full:
            H, dH = self.dH(par,full = True)
            AH = self.Ah(H)
            AD = Dk@Dv - AH
        else:
            dH = self.dH(par,full = False)
        dAD = []
        dAD.append(Dk@Dv)
        for i in range(len(dH)):
            dAD.append(- self.Ah(dH[i]))
        if full:
            return(AD,dAD)
        else:
            return(dAD)
        
    def getPars(self) -> np.ndarray:
        return(np.hstack([self.kappa,self.gamma, self.vx, self.vy]))
    
    def setPars(self,par)-> None:
        self.kappa = par[0]
        self.gamma = par[1]
        self.vx = par[2]
        self.vy = par[3]
        
    def setClib(self):
        if not os.path.exists('./ccode/lib_AcH_2D_b%d.so'%(self.bc)):
            os.system('g++ -c -fPIC ./ccode/AcH_2D_b%d.cpp -o ./ccode/AcH_2D_b%d.o'%(self.bc,self.bc))
            os.system('g++ -shared -o ./ccode/lib_AcH_2D_b%d.so ./ccode/AcH_2D_b%d.o'%(self.bc,self.bc))
            os.system('rm ./ccode/AcH_2D_b%d.o'%(self.bc))
        self.lib = ctypes.cdll.LoadLibrary('./ccode/lib_AcH_2D_b%d.so'%(self.bc))
        self.AHnew = self.lib.AH_new
        self.AHnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,shape = (2,2)), ctypes.c_double,ctypes.c_double]
        self.AHnew.restype = ctypes.c_void_p
        self.AHrow = self.lib.AH_Row
        self.AHrow.argtypes = [ctypes.c_void_p]
        self.AHrow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.grid.M*self.grid.N*9,))
        self.AHcol = self.lib.AH_Col
        self.AHcol.argtypes = [ctypes.c_void_p]
        self.AHcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (self.grid.M*self.grid.N*9,))
        self.AHval = self.lib.AH_Val
        self.AHval.argtypes = [ctypes.c_void_p]
        self.AHval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (self.grid.M*self.grid.N*9,))
        self.AHdel = self.lib.AH_delete
        self.AHdel.argtypes = [ctypes.c_void_p]
        self.AHdel.restype = None
    
        
    def Ah(self,H = None) -> sparse.csc_matrix:
        obj = self.AHnew(self.grid.M, self.grid.N, H, self.grid.hx, self.grid.hy)
        row = self.AHrow(obj)
        col = self.AHcol(obj)
        val = self.AHval(obj)

        rem = row != (self.grid.M*self.grid.N)
        row = row[rem]
        col = col[rem]
        val = val[rem]
        res = sparse.csc_matrix((val, (row, col)), shape=(self.grid.M*self.grid.N, self.grid.M*self.grid.N))
        self.AHdel(obj)
        return(res)
    
    def H(self,par = None) -> np.ndarray:
        return(np.exp(par[1])*np.eye(2) + par[2:4][:,np.newaxis]*par[2:4][np.newaxis,:])
    
    def dH(self,par,full = False):
        gamma = np.exp(par[1])
        vv = par[2:4]
        H_gamma = gamma*np.eye(2) 
        dv = np.array([1,0])
        H_vx = dv[:,np.newaxis]*vv[np.newaxis,:] + vv[:,np.newaxis]*dv[np.newaxis,:]
        dv = np.array([0,1])
        H_vy = dv[:,np.newaxis]*vv[np.newaxis,:] + vv[:,np.newaxis]*dv[np.newaxis,:]
        if full:
            H = gamma*np.eye(2) + vv[:,np.newaxis]*vv[np.newaxis,:]
            return(H,(H_gamma,H_vx,H_vy))
        else:
            return((H_gamma,H_vx,H_vy))

    def print(self,par):
        return("\u03BA = %2.2f"%(par[0]) +  ", \u03B3 = %2.2f"%(par[1]) + ", vx = %2.2f"%(par[2]) + ", vy = %2.2f"%(par[3]))
    
    
    def plot(self):
        return


# pg = np.exp(self.grid.evalBH(par = par[0]))
#         pvx = self.grid.evalBH(par = vx)
#         pvy = self.grid.evalBH(par = vy)
#         aV = np.sqrt(pvx**2 + pvy**2)
#         mV = np.array([[pvx,pvy],[pvy,-pvx]]).T.swapaxes(0,1)
#         cosh_aV = (np.exp(aV)+ np.exp(-aV))/2
#         sinh_aV = (np.exp(aV)- np.exp(-aV))/2