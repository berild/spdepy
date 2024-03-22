import numpy as np
from scipy import sparse
import ctypes
import os
from sksparse.cholmod import cholesky

class CovAdvectionVarDiffusion2D:
    """AdvectionDiffusion _summary_
    
    Parameters = log kappa^2, log gamma, vx, vy, lambda log sigma, log tau
    """
    def __init__(self,grid,mod0,par=None,bc = 3, ww = None) -> None:
        self.grid = grid
        self.type = "cov-advection-var-diffusion-2D-bc%d"%(bc)
        self.Q = None
        self.Q_fac = None
        self.data = None
        self.r = None
        self.S = None
        self.bc = bc
        self.ww = ww
        self.mod0 = mod0
        self.AHnew = None
        self.Awnew = None
        self.fitQ0 = True
        if par is None:
            par = np.hstack([[-1]*9,[-1]*9,[0.1]*9,[0.1]*9,0.1,0,np.log(100)],dtype="float64")
        self.setPars(par)
    
    def getPars(self,onlySelf = True) -> np.ndarray:
        if onlySelf:
            return(np.hstack([self.kappa,self.gamma,self.vx,self.vy,self.lamb,self.sigma,self.tau],dtype = "float64"))
        else:
            return(np.hstack([self.kappa,self.gamma,self.vx,self.vy,self.lamb,self.sigma,self.mod0.getPars()[:-1],self.tau],dtype = "float64"))
    
    def setPars(self,par)-> None:
        par = np.array(par,dtype = "float64")
        self.kappa = par[0:9]
        self.gamma = par[9:18]
        self.vx = par[18:27]
        self.vy = par[27:36]
        self.lamb = par[36]
        self.sigma = par[37]
        if par.size > 39:
           self.mod0.setPars(par[38:]) 
        self.tau = par[-1]
        
    def initFit(self,data,**kwargs):
        assert data.shape[0] <= self.grid.n 
        assert kwargs.get("ww") is not None or self.ww is not None
        self.ww = kwargs.get("ww") if kwargs.get("ww") is not None else self.ww
        if kwargs.get("fitQ0") is not None:
            assert type (kwargs.get("fitQ0")) == bool
            self.fitQ0 = kwargs.get("fitQ0")
        if self.fitQ0:
            x0 = self.mod0.getPars()[:-1]
            par = np.hstack([[-1]*9,[-1]*9,[0.1]*9,[0.1]*9,0.1,0,x0,np.log(100)],dtype="float64")
        else:
            par = np.hstack([[-1]*9,[-1]*9,[0.1]*9,[0.1]*9,0.1,0,np.log(100)],dtype="float64")
        self.dA_w = self.Aw(self.ww)
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
        self.Q, self.Q_fac,_ = self.makeQ(par = par, grad = False)
        self.S = self.grid.getS()
    
    def print(self,par):
        if par.size > 39:
            s1 ="| \u03BA = %2.2f"%(np.exp(par[0:9]).mean()) +  ", \u03B3 = %2.2f"%(np.exp(par[9:18]).mean()) + ", vx = %2.2f"%(par[18:27].mean()) + ", vy = %2.2f"%(par[27:36].mean()) +  ", \u039B = %2.2f"%(np.exp(par[36])) +", \u03C3 = %2.2f"%(np.exp(par[37])) +  ", \u03C4 = %2.2f\n"%(np.exp(par[-1])) 
            s1 += " Q0: " + self.mod0.print(par[38:])[:-10]
            return s1
        else:
            return("| \u03BA = %2.2f"%(np.exp(par[0:9]).mean()) +  ", \u03B3 = %2.2f"%(np.exp(par[9:18]).mean()) + ", vx = %2.2f"%(par[18:27].mean()) + ", vy = %2.2f"%(par[27:36].mean()) +  ", \u039B = %2.2f"%(np.exp(par[36])) +", \u03C3 = %2.2f"%(np.exp(par[37])) +  ", \u03C4 = %2.2f"%(np.exp(par[38])))

    def makeQ(self, par, grad = True):
        assert self.ww is not None
        # grid
        dt = self.grid.dt
        T = self.grid.T
        Ns = self.grid.Ns
        Dv = self.grid.Dv
        iDv = self.grid.iDv
        # parameters
        kappa = np.exp(self.grid.evalB(par[0:9]))
        gamma = np.exp(self.grid.evalBH(par[9:18]))
        vx = self.grid.evalBH(par[18:27])
        vy = self.grid.evalBH(par[27:36])
        vv = np.stack([vx,vy],axis = 2)
        sigma = np.exp(par[37])
        Dk = sparse.diags(kappa).tocsc()
        ws = par[36]*self.ww
        # components
        Hs = (np.eye(2)*(np.stack([gamma,gamma],axis=2))[:,:,:,np.newaxis]) + vv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
        As = Dv@Dk
        Qs = As.transpose()@iDv@As
        A = Dv + (Dv@Dk - self.Ah(Hs) +  self.Aw(ws))*dt
        if par.size > 39:
            Q0,_,dQ0 = self.mod0.makeQ(par[38:],grad = grad)
        else:
            if self.mod0.Q is None:
                self.mod0.setQ(par[38:])
            Q0 = self.mod0.Q
        # precision matrix Q
        Q = sparse.bmat([[sigma*dt*Q0 + Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-2)*Ns))]])
        for t in range(T-2):
            Q = sparse.bmat([[Q],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A + Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
        Q = sparse.bmat([[Q],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A]])]])
        Q = 1/(dt*sigma)*Q.tocsc()
        Q_fac = cholesky(Q)
        # gradient
        if grad:
            dQ = []
            # log kappa 2
            for i in range(9):
                dA = dt*Dv@(sparse.diags(self.grid.bs[:,i]*kappa).tocsc())
                dAs = Dv@(sparse.diags(self.grid.bs[:,i]*kappa).tocsc())
                dQs = dAs.T@iDv@As + As.T@iDv@dAs
                tdQ = sparse.bmat([[dQs,-dQs@iDv@A - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
                for t in range(T-2):
                    tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs - A.T@iDv@dQs, dA.T@iDv@Qs@iDv@A + A.T@iDv@dQs@iDv@A + A.T@iDv@Qs@iDv@dA + dQs, -dQs@iDv@A - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs - A.T@iDv@dQs, dA.T@iDv@Qs@iDv@A + A.T@iDv@dQs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
                dQ.append(tdQ.tocsc()/(dt*sigma))
            # log gamma
            for i in range(9):
                dHs = np.eye(2)*(np.stack([self.grid.bsH[:,:,i]*gamma,self.grid.bsH[:,:,i]*gamma],axis=2)[:,:,:,np.newaxis])
                dA = - self.Ah(dHs)*dt
                tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
                for t in range(T-2):
                    tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs , dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
                dQ.append(tdQ.tocsc()/(dt*sigma))
            # vx
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dv = np.stack([self.grid.evalBH(par = dpar),self.grid.evalBH(par = np.zeros(9))],axis = 2)
                dHs = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
                dA = - self.Ah(dHs)*dt
                tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
                for t in range(T-2):
                    tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs , dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
                dQ.append(tdQ.tocsc()/(dt*sigma))
            # vy
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dv = np.stack([self.grid.evalBH(par = np.zeros(9)),self.grid.evalBH(par = dpar)],axis = 2)
                dHs = vv[:,:,:,np.newaxis]*dv[:,:,np.newaxis,:]  + dv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:]
                dA = - self.Ah(dHs)*dt
                tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
                for t in range(T-2):
                    tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs , dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
                dQ.append(tdQ.tocsc()/(dt*sigma))
            # lambda
            dA = self.dA_w*dt
            tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
            for t in range(T-2):
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
            tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
            dQ.append(1/(dt*sigma)*tdQ.tocsc())
            # log sigma 2
            tdQ = sparse.bmat([[Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-2)*Ns))]])
            for t in range(T-2):
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A + Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
            tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A]])]])
            dQ.append(-1/(dt*sigma)*tdQ.tocsc())
            if par.size > 39:
                for j in range(len(dQ0)): 
                    tdQ = sparse.bmat([[dQ0[j], sparse.csc_matrix((Ns,(T-1)*Ns))]])
                    for t in range(T-1):
                            tdQ = sparse.bmat([[tdQ],[sparse.csc_matrix((Ns,(T)*Ns))] ])
                    dQ.append((tdQ).tocsc())
            return(Q,Q_fac,dQ)
        else:
            return(Q,Q_fac,None)
 
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
            Q, Q_fac,_ = self.makeQ(par = par, grad = False)
            Q_c = Q + self.S.T@self.S*tau
            Q_c_fac= cholesky(Q_c)
            if (Q_fac == -1) or (Q_c_fac == -1):
                return(self.like)
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            if self.r == 1:
                data = data.reshape(-1,1)
                mu_c = mu_c.reshape(-1,1)
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*np.log(tau)/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - tau/2*((data-self.S@mu_c)**2).sum()
            like =  -like/(self.S.shape[0]*self.r)
            return(like)
        
    def Aw(self,ws, diff = 3) -> sparse.csc_matrix:
        if self.Awnew is None:
            self.setClib()
        M,N,T = self.grid.shape
        ws = np.array(ws,dtype = "float64")
        obj = self.Awnew(M, N, ws, self.grid.hx, self.grid.hy, diff,np.zeros((M*N,4),dtype = "float64"))
        row = self.Awrow(obj)
        col = self.Awcol(obj)
        val = self.Awval(obj)
        rem = row != (M*N)
        row = row[rem]
        col = col[rem]
        val = val[rem]
        res = sparse.csc_matrix((val, (row, col)), shape=(M*N, M*N))
        self.Awdel(obj)
        return(res)
    
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
        
    def setClib(self) -> None:
        tmp = os.path.dirname(__file__)
        if not os.path.exists(tmp + '/ccode/lib_Aw_2D_b%d.so'%(self.bc)):
            os.system('g++ -c -fPIC %s/ccode/Aw_2D_b%d.cpp -o %s/ccode/Aw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            import platform
            if  'Windows' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_Aw_2D_b%d.so -o %s/ccode/lib_Aw_2D_b%d.so %s/ccode/Aw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc)) # not tested
            elif  'Linux' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_Aw_2D_b%d.so -o %s/ccode/lib_Aw_2D_b%d.so %s/ccode/Aw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc))
            else:
                os.system('g++ -shared -o %s/ccode/lib_Aw_2D_b%d.so %s/ccode/Aw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            os.system('rm %s/ccode/Aw_2D_b%d.o'%(tmp,self.bc))
        self.libAw = ctypes.cdll.LoadLibrary('%s/ccode/lib_Aw_2D_b%d.so'%(tmp,self.bc))
        M,N,T= self.grid.shape
        self.Awnew = self.libAw.Aw_new
        self.Awnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,shape = (M*N,4)), ctypes.c_double,ctypes.c_double,ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,shape = (M*N,4))]
        self.Awnew.restype = ctypes.c_void_p
        self.Awrow = self.libAw.Aw_Row
        self.Awrow.argtypes = [ctypes.c_void_p]
        self.Awrow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*5,))
        self.Awcol = self.libAw.Aw_Col
        self.Awcol.argtypes = [ctypes.c_void_p]
        self.Awcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*5,))
        self.Awval = self.libAw.Aw_Val
        self.Awval.argtypes = [ctypes.c_void_p]
        self.Awval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (M*N*5,))
        self.Awdel = self.libAw.Aw_delete
        self.Awdel.argtypes = [ctypes.c_void_p]
        self.Awdel.restype = None
        
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