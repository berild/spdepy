import numpy as np
from scipy import sparse
import ctypes
import os
from sksparse.cholmod import cholesky

class VarWhittleMaternAnisotropic2D:
    """ Whittle Matern _summary_
    
    Parameters = log kappa^2, log gamma, vx, vy, log tau
    """
    def __init__(self,grid,par=None,bc = 3) -> None:
        self.grid = grid
        self.type = "var-whittle-matern-anisotropic-2D-bc%d"%(bc)
        self.Q = None
        self.Q_fac = None
        self.data = None
        self.r = None
        self.S = None
        self.bc = bc
        self.AHnew = None
        self.Awnew = None
        if par is None: 
            par = np.hstack([[-1]*9,[-1]*9, [0.1]*9, [0.1]*9,np.log(100)],dtype = "float64")
            self.setPars(par)
        else:
            self.setQ(par = par)
    
    def getPars(self,*args, **kwargs) -> np.ndarray:
        return(np.hstack([self.kappa,self.gamma,self.vx,self.vy,self.tau],dtype = "float64"))
    
    def setPars(self,par) -> None:
        par = np.array(par,dtype="float64")
        self.kappa = par[0:9]
        self.gamma = par[9:18]
        self.vx = par[18:27]
        self.vy = par[27:36]
        self.tau = par[36]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))

    def initFit(self,data, **kwargs):
        par = np.hstack([[-1]*9,[-1]*9, [0.1]*9, [0.1]*9,np.log(100)],dtype = "float64")
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
        self.Q, self.Q_fac, _ = self.makeQ(par = par, grad = False)
        self.S = self.grid.getS()
        # Hs = self.getH()
        # Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        # A_mat = self.grid.Dv@Dk - self.Ah(Hs)
        # self.Q = A_mat.T@self.grid.iDv@A_mat
        # self.S = self.grid.getS()

    def makeQ(self, par, grad = True):
        Ns = self.grid.Ns
        Dv = self.grid.Dv
        iDv = self.grid.iDv
        # parameters
        kappa = np.exp(self.grid.evalB(par = par[0:9]))
        gamma = np.exp(self.grid.evalBH(par = par[9:18]))
        vx = self.grid.evalBH(par[18:27])
        vy = self.grid.evalBH(par[27:36])
        vv = np.stack([vx,vy],axis=2)
        # components
        Hs = (np.eye(2)*(np.stack([gamma,gamma],axis=2))[:,:,:,np.newaxis]) + vv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
        Dk = sparse.diags(kappa).tocsc()
        Ah = self.Ah(Hs)
        A = Dv@Dk - Ah
        # Precision matrix
        Q = A.transpose()@iDv@A
        Q_fac = cholesky(Q)
        # gradient
        if grad: 
            dQ = []
            for i in range(9):
                dA = Dv@sparse.diags(self.grid.bs[:,i]*kappa)
                dQ.append((dA.transpose()@iDv@A +  A.transpose()@iDv@dA).tocsc())
            # log gamma
            for i in range(9):
                dHs = np.eye(2)*(np.stack([self.grid.bsH[:,:,i]*gamma,self.grid.bsH[:,:,i]*gamma],axis=2)[:,:,:,np.newaxis])
                dA = - self.Ah(dHs)
                dQ.append((dA.transpose()@iDv@A +  A.transpose()@iDv@dA).tocsc())
            # vx
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dv = np.stack([self.grid.evalBH(par = dpar),self.grid.evalBH(par = np.zeros(9))],axis = 2)
                dHs = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
                dA = - self.Ah(dHs)
                dQ.append((dA.transpose()@iDv@A +  A.transpose()@iDv@dA).tocsc())
            # vy
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dv = np.stack([self.grid.evalBH(par = np.zeros(9)),self.grid.evalBH(par = dpar)],axis = 2)
                dHs = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:] 
                dA = - self.Ah(dHs)
                dQ.append((dA.transpose()@iDv@A +  A.transpose()@iDv@dA).tocsc())
            return(Q,Q_fac,dQ)
        else:
            return(Q,Q_fac,None)
        
    # def getH(self,gamma=None,vx = None,vy = None,d=None,grad = False):
    #     if vx is None and vy is None and gamma is None:
    #         gamma = self.gamma
    #         vx = self.vx
    #         vy = self.vy
    #     if not grad:
    #         pg = np.exp(self.grid.evalBH(par = gamma))
    #         vv = np.stack([self.grid.evalBH(par = vx),self.grid.evalBH(par = vy)],axis=2)
    #         H = (np.eye(2)*(np.stack([pg,pg],axis=2))[:,:,:,np.newaxis]) + vv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
    #         return(H)
    #     else:
    #         dpar = np.zeros(9)
    #         dpar[d] = 1

    #         vv = np.stack([self.grid.evalBH(par = vx),self.grid.evalBH(par = vy)],axis=2)
    #         pg = np.exp(self.grid.evalBH(par = gamma))
    #         H_gamma = np.eye(2)*(np.stack([self.grid.bsH[:,:,d]*pg,self.grid.bsH[:,:,d]*pg],axis=2)[:,:,:,np.newaxis])

    #         dv = np.stack([self.grid.evalBH(par = dpar),self.grid.evalBH(par = np.zeros(9))],axis = 2)
    #         H_vx = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]

    #         dv = np.stack([self.grid.evalBH(par = np.zeros(9)),self.grid.evalBH(par = dpar)],axis = 2)
    #         H_vy = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:] 
    #         return((H_gamma,H_vx,H_vy))

    def print(self,par):
        return("| \u03BA = %2.2f"%(np.exp(par[0:9]).mean()) +  ", \u03B3 = %2.2f"%(np.exp(par[9:18]).mean()) + ", vx = %2.2f"%(par[18:27].mean()) + ", vy = %2.2f"%(par[27:36].mean()) +  ",\u03C4 = %2.2f"%(np.exp(par[36])))
    
    def logLike(self, par, nh1 = 100, grad = True):
        if grad:
            data  = self.data
            tau = np.exp(par[-1])
            Q, Q_fac, dQ = self.makeQ(par = par, grad = True)
            Q_c = Q + self.S.T@self.S*tau
            Q_c_fac = cholesky(Q_c)
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
            Q, Q_fac, _ = self.makeQ(par = par, grad = False)
            Q_c = Q + self.S.T@self.S*tau
            Q_c_fac= cholesky(Q_c)
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            if self.r == 1:
                data = data.reshape(-1,1)
                mu_c = mu_c.reshape(-1,1)
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*np.log(tau)/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - tau/2*((data-self.S@mu_c)**2).sum()
            like =  -like/(self.S.shape[0]*self.r)
            return(like)

    # def logLike(self, par, nh1 = 100,grad = True):
    #     #mod4: kappa(0:9), gamma(9:18), vx(18:27), vy(27:36), sigma(36)
    #     data  = self.data
    #     Hs = self.getH(gamma = par[9:18],vx = par[18:27], vy = par[27:36]) 
    #     lkappa = self.grid.evalB(par = par[0:9])
    #     Dk =  sparse.diags(np.exp(lkappa)) 
    #     Dv = self.grid.Dv
    #     iDv = self.grid.iDv
    #     A_mat = Dv@Dk - self.Ah(Hs)
    #     Q = A_mat.transpose()@iDv@A_mat
    #     Q_c = Q + self.S.transpose()@self.S*np.exp(par[36])
    #     Q_fac = cholesky(Q)
    #     Q_c_fac= cholesky(Q_c)
    #     if (Q_fac == -1) or (Q_c_fac == -1):
    #         if grad:
    #             return((self.like,self.jac))
    #         else:
    #             return(self.like)
    #     mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[36]))
    #     if self.r == 1:
    #         data = data.reshape(-1,1)
    #         mu_c = mu_c.reshape(-1,1)
    #     like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[36]/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - np.exp(par[36])/2*((data-self.S@mu_c)**2).sum()
    #     if grad:
    #         vtmp = (2*np.random.randint(1,3,self.grid.Ns*nh1)-3).reshape(self.grid.Ns,nh1)
    #         TrQ = Q_fac.solve_A(vtmp)
    #         TrQc = Q_c_fac.solve_A(vtmp)
    #         g_par = np.zeros(37)
            
    #         g_par[36] = self.S.shape[0]*self.r/2 - 1/2*(TrQc*(self.S.transpose()@self.S*np.exp(par[36])@vtmp)).sum()*self.r/nh1 - np.exp(par[36])/2*((data - self.S@mu_c)**2).sum()

    #         for i in range(9):
    #             A_par = Dv@sparse.diags(self.grid.bs[:,i]*np.exp(lkappa))
    #             Q_par = A_par.transpose()@iDv@A_mat + A_mat.transpose()@iDv@A_par
    #             dQmu_c = Q_par@mu_c
    #             g_par[i] =  1/2*((TrQ - TrQc)*(Q_par@vtmp)).sum()*self.r/nh1 - 1/2*(mu_c*(dQmu_c)).sum()

    #             dH = self.getH(gamma = par[9:18],vx = par[18:27],vy = par[27:36], d=i,grad=True) 
    #             for j in range(3):
    #                 A_par = - self.Ah(dH[j])
    #                 Q_par = A_par.transpose()@iDv@A_mat +  A_mat.transpose()@iDv@A_par
    #                 dQmu_c = Q_par@mu_c
    #                 g_par[9*(j+1) + i] = 1/2*((TrQ - TrQc)*(Q_par@vtmp)).sum()*self.r/nh1 - 1/2*(mu_c*(dQmu_c)).sum()
    #         jac =  -g_par/(self.S.shape[0]*self.r)
    #     like =  -like/(self.S.shape[0]*self.r)
    #     if grad: 
    #         return((like,jac))
    #     else:
    #         return(like)
    
    def Ah(self,Hs) -> sparse.csc_matrix:
        if self.AHnew is None:
            self.setClib()
        M, N = self.grid.shape
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
        M, N = self.grid.shape
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