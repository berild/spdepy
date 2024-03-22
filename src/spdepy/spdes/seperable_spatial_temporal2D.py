import numpy as np
from scipy import sparse
import ctypes
import os
from sksparse.cholmod import cholesky

class SeperableSpatialTemporal2D:
    """ Whittle Matern _summary_
    
    Parameters = log kappa^2, log gamma, vx, vy, a, log tau_omega, log tau
    """
    def __init__(self,grid,par=None,bc = 3) -> None:
        self.grid = grid
        self.type = "seperable-spatial-temporal-ani-2D-bc%d"%(bc)
        self.Q = None
        self.Q_fac = None
        self.data = None
        self.r = None
        self.S = None
        self.bc = bc
        self.AHnew = None
        if par is None: 
            par = np.hstack([[-1]*9,[-1]*9, [0.1]*9, [0.1]*9,0.1,1,np.log(100)],dtype = "float64")
            self.setPars(par)
        else:
            self.setQ(par = par)
    
    def getPars(self,*args, **kwargs) -> np.ndarray:
        return(np.hstack([self.kappa,self.gamma,self.vx,self.vy,self.a, self.sigma,self.tau],dtype = "float64"))
    
    def setPars(self,par) -> None:
        par = np.array(par,dtype="float64")
        self.kappa = par[0:9]
        self.gamma = par[9:18]
        self.vx = par[18:27]
        self.vy = par[27:36]
        self.a = par[36]
        self.sigma = par[37]
        self.tau = par[38]

    def initFit(self,data, **kwargs):
        assert data.shape[0] <= self.grid.n
        par = np.hstack([[-1]*9,[-1]*9, [0.1]*9, [0.1]*9,0.1,1,np.log(100)],dtype = "float64")
        self.data = data
        if self.data.ndim == 2:
            self.r = self.data.shape[1]
        else:
            self.r = 1
        self.S = self.grid.getS(idxs = kwargs.get("idx"))
        return(par)

    def setQ(self,par = None,S = None):
        if par is None:
            par = self.getPars()
        else:
            self.setPars(par)
        if S is not None:
            self.S = S
        self.Q, self.Q_fac = self.makeQ(par = par, grad = False)
        self.S = self.grid.getS()
    
    def print(self,par):
        return("| \u03BA = %2.2f"%(np.exp(par[0:9]).mean()) +  ", \u03B3 = %2.2f"%(np.exp(par[9:18]).mean()) + ", vx = %2.2f"%((par[18:27]).mean()) + ", vy = %2.2f"%((par[27:36]).mean()) + ", a = %2.2f"%(par[36]) + ", \u03C3 = %2.2f"%(np.exp(par[37])) +  ", \u03C4 = %2.2f"%(np.exp(par[38])))

    def makeQ(self, par, grad = True):
        # grid
        T = self.grid.T
        Dv = self.grid.Dv
        iDv = self.grid.iDv
        # parameters
        kappa = np.exp(self.grid.evalB(par[0:9]))
        gamma = np.exp(self.grid.evalBH(par[9:18]))
        vx = self.grid.evalBH(par[18:27])
        vy = self.grid.evalBH(par[27:36])
        vv = np.stack([vx,vy],axis = 2)
        apar = par[36]
        sigma = np.exp(par[37])
        # components
        Hs = (np.eye(2)*(np.stack([gamma,gamma],axis=2))[:,:,:,np.newaxis]) + vv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
        Dk =  sparse.diags(kappa).tocsc()
        As = Dv@Dk - self.Ah(Hs)
        Qs = As.T@iDv@As
        Qt = self.makeQt(apar,sigma, T = T)
        # precision matrix Q
        Q = sparse.kron(Qt,Qs).tocsc()
        Q_fac = cholesky(Q)
        # gradient
        if grad:
            dQ = []
            # log kappa 2
            for i in range(9):
                dAs = Dv@(sparse.diags(self.grid.bs[:,i]*kappa).tocsc())
                dQs = dAs.T@iDv@As + As.T@iDv@dAs
                dQ.append(sparse.kron(Qt,dQs).tocsc())
            # log gamma
            for i in range(9):
                dHs = np.eye(2)*(np.stack([self.grid.bsH[:,:,i]*gamma,self.grid.bsH[:,:,i]*gamma],axis=2)[:,:,:,np.newaxis])
                dAs = - self.Ah(dHs)
                dQs = dAs.T@iDv@As + As.T@iDv@dAs
                dQ.append(sparse.kron(Qt,dQs).tocsc())
            # vx
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dv = np.stack([self.grid.evalBH(par = dpar),self.grid.evalBH(par = np.zeros(9))],axis = 2)
                dHs = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
                dAs = - self.Ah(dHs)
                dQs = dAs.T@iDv@As + As.T@iDv@dAs
                dQ.append(sparse.kron(Qt,dQs).tocsc())
            # vy
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dv = np.stack([self.grid.evalBH(par = np.zeros(9)),self.grid.evalBH(par = dpar)],axis = 2)
                dHs = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
                dAs = - self.Ah(dHs)
                dQs = dAs.T@iDv@As + As.T@iDv@dAs
                dQ.append(sparse.kron(Qt,dQs).tocsc())
            # a 
            dQt = self.makeQt(apar,sigma, T = T, diff = 1)
            dQ.append(sparse.kron(dQt,Qs).tocsc())
            # sigma
            dQ.append(Q)
            return(Q,Q_fac,dQ)
        else:
            return(Q,Q_fac)
        
                
    def logLike(self, par, nh1 = 100, grad = True):
        if grad:
            data  = self.data
            tau = np.exp(par[-1])
            Q, Q_fac, dQ = self.makeQ(par = par, grad = True)
            Q_c = Q + self.S.T@self.S*tau
            Q_c_fac = cholesky(Q_c)
            if (Q_fac == -1) or (Q_c_fac == -1):
                return((self.like,self.jac))
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            if self.r == 1:
                data = data.reshape(-1,1)
                mu_c = mu_c.reshape(-1,1)
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*np.log(tau)/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - tau/2*((data-self.S@mu_c)**2).sum()
            
            vtmp = (2*np.random.randint(1,3,self.grid.n*nh1)-3).reshape(self.grid.n,nh1)
            TrQ = Q_fac.solve_A(vtmp)
            TrQc = Q_c_fac.solve_A(vtmp)
            g_par = np.zeros(par.size)
            for i in range(par.size-1):
                dQmu_c = dQ[i]@mu_c
                g_par[i]= ((1/2*((TrQ - TrQc)*(dQ[i]@vtmp)).sum()*self.r/nh1 - 1/2*(mu_c*(dQmu_c)).sum()))
            g_par[-1] = self.S.shape[0]*self.r/2 - 1/2*(TrQc*(self.S.T@self.S*tau@vtmp)).sum()*self.r/nh1 - tau/2*((data - self.S@mu_c)**2).sum()
            jac =  -g_par/(self.S.shape[0]*self.r)
            like =  -like/(self.S.shape[0]*self.r)
            return((like,jac))
        else:
            data  = self.data
            tau = np.exp(par[-1])
            Q, Q_fac = self.makeQ(par = par, grad = False)
            Q_c = Q + self.S.T@self.S*tau
            Q_c_fac= cholesky(Q_c)
            if (Q_fac == -1) or (Q_c_fac == -1):
                return(self.like)
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            if self.r == 1:
                data = data.reshape(-1,1)
                mu_c = mu_c.reshape(-1,1)
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*np.log(tau)/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - tau/2*((data-self.S@mu_c)**2).sum()
            like =  -like/(self.S.shape[0]*self.r)
            return(like)
    
    def Ah(self,Hs) -> sparse.csc_matrix:
        if self.AHnew is None:
            self.setClib()
        M, N, T = self.grid.shape
        Hs = np.array(Hs,dtype = "float64")
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
    
    def makeQt(self,a,sigma, T = 10, diff = 0) -> sparse.csc_matrix:
        res = np.zeros((T,T))
        if diff == 1:
            for i in range(T):
                if i == 0 or i == T-1:
                    res[i,i] = 0
                else:
                    res[i,i] = 2*a*sigma
                if i > 0:
                    res[i,i-1] = -sigma
                if i < T-1:
                    res[i,i+1] = -sigma
            return(sparse.csc_matrix(res))
        else:
            for i in range(T):
                if i == 0 or i == T-1:
                    res[i,i] = sigma
                else:
                    res[i,i] = (1+a**2)*sigma
                if i > 0:
                    res[i,i-1] = -a*sigma
                if i < T-1:
                    res[i,i+1] = -a*sigma
            return(sparse.csc_matrix(res))

        
    def setClib(self) -> None:
        tmp = os.path.dirname(__file__)
        M, N, T = self.grid.shape
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