import numpy as np
from scipy import sparse
import ctypes
import os
from sksparse.cholmod import cholesky

class WhittleMatern2D:
    """ Whittle Matern _summary_
    
    Parameters = log kappa^2, log gamma, log tau
    """
    def __init__(self,grid,par=None,bc = 3) -> None:
        self.grid = grid
        self.type = "whittle-matern-isotropic-2D-bc%d"%(bc)
        self.Q = None
        self.Q_fac = None
        self.data = None
        self.r = None
        self.S = None
        self.bc = bc
        self.AHnew = None
        self.Awnew = None
        if par is None: 
            par = np.hstack([[-1]*9,[-0.5]*9,1],dtype = "float64")
            self.setPars(par)
        else:
            self.setQ(par = par)
    
    def getPars(self) -> np.ndarray:
        return(np.hstack([self.kappa,self.gamma,self.tau],dtype = "float64"))
    
    def setPars(self,par) -> None:
        self.kappa = par[0:9]
        self.gamma = par[9:18]
        self.tau = par[18]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))

    def initFit(self,data, **kwargs):
        #mod4: kappa(0:9), gamma(9:18), sigma(18)
        assert data.shape[0] <= self.grid.Ns
        par = np.hstack([[-1]*9,[-0.5]*9,1],dtype = "float64")
        self.data = data
        if self.data.ndim == 2:
            self.r = self.data.shape[1]
        else:
            self.r = 1
        self.S = self.grid.getS()
        return(par)

    def sample(self,n = 1,simple = False):
        z = np.random.normal(size = self.grid.Ns*n).reshape(self.grid.Ns,n)
        data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False))
        if not simple:
            data += z*np.exp(self.sigma)
        return(data)


    def setQ(self,par = None,S = None, simple = False):
        if par is None:
            assert(self.kappa is not None and self.gamma is not None and self.sigma is not None)
            par = self.getPars()
        else:
            self.setPars(par)
        if S is not None:
            self.S = S
        Hs = self.getH()
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_mat = self.grid.Dv@Dk - self.Ah(Hs)
        self.Q = A_mat.T@self.grid.iDv@A_mat
        self.S = self.grid.getS()
        if not simple:
            self.Q_fac = cholesky(self.Q)

    def getH(self,gamma=None,d=None,grad = False):
        if gamma is None:
            gamma = self.gamma
        if not grad:
            pg = np.exp(self.grid.evalBH(par = gamma))
            H = (np.eye(2)*(np.stack([pg,pg],axis=2))[:,:,:,np.newaxis])
            return(H)
        else:
            dpar = np.zeros(9)
            dpar[d] = 1
            
            pg = np.exp(self.grid.evalBH(par = gamma))
            H_gamma = np.eye(2)*(np.stack([self.grid.bsH[:,:,d]*pg,self.grid.bsH[:,:,d]*pg],axis=2)[:,:,:,np.newaxis])
            return(H_gamma)
        
    def print(self,par):
        return("| \u03BA = %2.2f"%(np.exp(par[0:9]).mean()) +  ", \u03B3 = %2.2f"%(np.exp(par[9:18]).mean())  +  ",\u03C4 = %2.2f"%(np.exp(par[18])))
    
    def logLike(self, par, nh1 = 100,grad = True):
        #mod4: kappa(0:9), gamma(9:18), sigma(18)
        data  = self.data
        Hs = self.getH(gamma = par[9:18]) 
        lkappa = self.grid.evalB(par = par[0:9])
        Dk =  sparse.diags(np.exp(lkappa)) 
        Dv = self.grid.Dv
        iDv = self.grid.iDv
        A_mat = Dv@Dk - self.Ah(Hs)
        Q = A_mat.transpose()@iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[18])
        Q_fac = cholesky(Q)
        Q_c_fac= cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[18]))
        if self.r == 1:
            data = data.reshape(-1,1)
            mu_c = mu_c.reshape(-1,1)
        like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[18]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - np.exp(par[18])/2*((data-self.S@mu_c)**2).sum()
        if grad:
            vtmp = (2*np.random.randint(1,3,self.grid.Ns*nh1)-3).reshape(self.grid.Ns,nh1)
            TrQ = Q_fac.solve_A(vtmp)
            TrQc = Q_c_fac.solve_A(vtmp)
            g_par = np.zeros(19)
            
            g_par[18] = self.S.shape[0]*self.r/2 - 1/2*(TrQc*(self.S.transpose()@self.S*np.exp(par[18])@vtmp)).sum()*self.r/nh1 - np.exp(par[18])/2*((data - self.S@mu_c)**2).sum()

            for i in range(9):
                A_par = Dv@sparse.diags(self.grid.bs[:,i]*np.exp(lkappa))
                Q_par = A_par.transpose()@iDv@A_mat + A_mat.transpose()@iDv@A_par
                dQmu_c = Q_par@mu_c
                g_par[i] =  1/2*((TrQ - TrQc)*(Q_par@vtmp)).sum()*self.r/nh1 - 1/2*(mu_c*(dQmu_c)).sum()

                dH = self.getH(gamma = par[9:18], d=i,grad=True) 
                A_par = - self.Ah(dH)
                Q_par = A_par.transpose()@iDv@A_mat +  A_mat.transpose()@iDv@A_par
                dQmu_c = Q_par@mu_c
                g_par[9 + i] = 1/2*((TrQ - TrQc)*(Q_par@vtmp)).sum()*self.r/nh1 - 1/2*(mu_c*(dQmu_c)).sum()
            jac =  -g_par/(self.S.shape[0]*self.r)
        like =  -like/(self.S.shape[0]*self.r)
        if grad: 
            return((like,jac))
        else:
            return(like)
    
    def Ah(self,Hs) -> sparse.csc_matrix:
        if self.AHnew is None:
            self.setClib()
        Hs = np.array(Hs,dtype = "float64")
        M, N = self.grid.shape
        obj = self.AHnew(M, N, Hs, self.grid.hx, self.grid.hy)
        row = self.AHrow(obj)
        col = self.AHcol(obj)
        val = self.AHval(obj)

        rem = row != (M*N)
        row = row[rem]
        col = col[rem]
        val = val[rem]
        res = sparse.csc_matrix((val, (row, col)), shape=(M*N, M*N))
        self.AHdel(obj)
        return(res)
        
    def setClib(self) -> None:
        tmp = os.path.dirname(__file__)
        if not os.path.exists(tmp + '/ccode/lib_AH_2D_b%d.so'%(self.bc)):
            os.system('g++ -c -fPIC %s/ccode/AH_2D_b%d.cpp -o %s/ccode/AH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            import platform
            if  'Windows' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_AH_2D_b%d.so -o %s/ccode/lib_AH_2D_b%d.so %s/ccode/AH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc)) # not tested
            elif  'Linux' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_AH_2D_b%d.so -o %s/ccode/lib_AH_2D_b%d.so %s/ccode/AH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc))
            else:
                os.system('g++ -shared -o %s/ccode/lib_AH_2D_b%d.so %s/ccode/AH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            os.system('rm %s/ccode/AH_2D_b%d.o'%(tmp,self.bc))
        self.libAh = ctypes.cdll.LoadLibrary('%s/ccode/lib_AH_2D_b%d.so'%(tmp,self.bc))
        self.AHnew = self.libAh.AH_new
        M, N = self.grid.shape
        self.AHnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=4,shape = (M*N,4,2,2)), ctypes.c_double,ctypes.c_double]
        self.AHnew.restype = ctypes.c_void_p
        self.AHrow = self.libAh.AH_Row
        self.AHrow.argtypes = [ctypes.c_void_p]
        self.AHrow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*9,))
        self.AHcol = self.libAh.AH_Col
        self.AHcol.argtypes = [ctypes.c_void_p]
        self.AHcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*9,))
        self.AHval = self.libAh.AH_Val
        self.AHval.argtypes = [ctypes.c_void_p]
        self.AHval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (M*N*9,))
        self.AHdel = self.libAh.AH_delete
        self.AHdel.argtypes = [ctypes.c_void_p]
        self.AHdel.restype = None
