import numpy as np
from scipy import sparse
import os
import ctypes
from sksparse.cholmod import cholesky

class AdvectionVarHaDiffusion2D:
    def __init__(self,grid,par=None,bc = 3, Q0 = None) -> None:
        self.grid = grid
        self.type = "advection-var-ha-diffusion-2D-bc%d"%(bc)
        self.Q = None
        self.Q_fac = None
        self.data = None
        self.r = None
        self.S = None
        self.Q0 = Q0
        self.bc = bc
        self.AHnew = None
        self.Awnew = None
        if par is None:
            par = np.hstack([[-1]*9,[1]*9,[1]*9,[1]*9,1,1,1,1])
            self.setPars(par)
        else:
            self.setQ(par = par)
    
    def getPars(self):
        return(np.hstack([self.kappa,self.gamma,self.vx,self.vy,self.wx,self.wy, self.sigma,self.tau]))
    
    def setPars(self,par)-> None:
        self.kappa = par[0:9]
        self.gamma = par[9:18]
        self.vx = par[18:27]
        self.vy = par[27:36]
        self.wx = par[36]
        self.wy = par[37]
        self.sigma = par[38]
        self.tau = par[39]
        
    def transDiff(self,par = None):
        if par is None:
            par = self.getPars()
        pg = np.exp(self.grid.evalB(par[9:18]))
        pvx = self.grid.evalB(par[18:27])
        pvy = self.grid.evalB(par[27:36])
        aV = np.sqrt(pvx**2 + pvy**2)
        cosh_aV = (np.exp(aV)+ np.exp(-aV))/2
        sinh_aV = (np.exp(aV)- np.exp(-aV))/2
        self.tgamma = pg*(cosh_aV-sinh_aV)
        self.tvx = np.sqrt(pg*sinh_aV/aV*(pvx + aV))
        self.tvy = np.sqrt(pg*sinh_aV/aV*(-pvx + aV))
        
    def initFit(self,data, **kwargs):
        assert data.shape[0] <= self.grid.n 
        assert kwargs.get("Q0") is not None or self.Q0 is not None
        self.Q0 = kwargs.get("Q0") if kwargs.get("Q0") is not None else self.Q0
        par = np.hstack([[-1]*9,[1]*9,[1]*9,[1]*9,1,1,1,1])
        self.data = data
        if self.data.ndim == 2:
            self.r = self.data.shape[1]
        else:
            self.r = 1
        self.S = self.grid.getS()
        self.Q0 = kwargs.get("Q0")
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
    
    def sample(self,n = 1,simple = False):
        z = np.random.normal(size = self.grid.n*n).reshape(self.grid.n,n)
        data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
        if not simple:
            data += z*1/np.sqrt(np.exp(self.tau))
        return(data)


    def setQ(self,par = None,S = None) -> None:
        if par is None:
            par = self.getPars()
        else:
            self.setPars(par)
        if S is not None:
            self.S = S
        self.Q, self.Q_fac = self.makeQ(par = par, grad = False)
    
    def print(self,par):
        return("| \u03BA = %2.2f"%(np.exp(par[0:9]).mean()) +  ", \u03B3 = %2.2f"%(np.exp(par[9:18]).mean()) + ", vx = %2.2f"%((par[18:27]).mean()) + ", vy = %2.2f"%((par[27:36]).mean()) + ", wx = %2.2f"%(par[36]) + ", wy = %2.2f"%(par[37]) + ", \u03C3 = %2.2f"%(np.exp(par[38])) +  ", \u03C4 = %2.2f"%(np.exp(par[39])))

    def makeQ(self, par, grad = True):
        assert self.Q0 is not None
        kappa = np.exp(self.grid.evalB(par[0:9]))
        gamma = np.exp(self.grid.evalBH(par[9:18]))
        vx = self.grid.evalBH(par[18:27])
        vy = self.grid.evalBH(par[27:36])
        aV = np.sqrt(vx**2 + vy**2)
        mV = np.array([[vx,vy],[vy,-vx]]).T.swapaxes(0,1)
        cosh_aV = (np.exp(aV) + np.exp(-aV))/2
        sinh_aV = (np.exp(aV) - np.exp(-aV))/2
        ws = par[36:38]
        sigma = np.exp(par[38])
        dt = self.grid.dt
        T = self.grid.T
        Ns = self.grid.Ns
        Dv = self.grid.Dv
        iDv = self.grid.iDv
        Hs = (gamma*cosh_aV)[:,:,np.newaxis,np.newaxis]*np.eye(2) + (gamma*sinh_aV/aV)[:,:,np.newaxis,np.newaxis]*mV
        Dk =  sparse.diags(kappa).tocsc()
        As = Dv@Dk
        Qs = As.transpose()@iDv@As
        A = Dv + Dv@Dk*dt - self.Ah(Hs)*dt + self.Aw(ws)*dt
        Q = sparse.bmat([[sigma*dt*self.Q0 + Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-2)*Ns))]])
        for t in range(T-2):
            Q = sparse.bmat([[Q],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A + Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
        Q = sparse.bmat([[Q],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A]])]])
        Q = 1/(dt*sigma)*Q.tocsc()
        Q_fac = cholesky(Q)
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
                dQ.append(tdQ/(dt*sigma))
            # log gamma
            for i in range(9):
                dHs = np.eye(2)*(np.stack([self.grid.bsH[:,:,i]*gamma,self.grid.bsH[:,:,i]*gamma],axis=2)[:,:,:,np.newaxis])
                dA = - self.Ah(dH)*dt
                tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
                for t in range(T-2):
                    tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs , dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
                dQ.append(tdQ/(dt*sigma))
            # vx
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dvx = self.grid.evalBH(par = dpar)
                dmV = np.array([[dvx,vy*0],[vy*0,-dvx]]).T.swapaxes(0,1)
                dH = (gamma*dvx*vx/aV)[:,:,np.newaxis,np.newaxis]*(sinh_aV[:,:,np.newaxis,np.newaxis]*np.eye(2)+ ((cosh_aV - sinh_aV/aV)/aV)[:,:,np.newaxis,np.newaxis]*mV)  + (gamma*sinh_aV/aV)[:,:,np.newaxis,np.newaxis]*dmV
                dA = - self.Ah(dH)*dt
                tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
                for t in range(T-2):
                    tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs , dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
                dQ.append(tdQ/(dt*sigma))
            # vy
            for i in range(9):
                dpar = np.zeros(9)
                dpar[i] = 1
                dvy = self.grid.evalBH(par = dpar)
                dmV = np.array([[vx*0,dvy],[dvy,-vx*0]]).T.swapaxes(0,1)
                dH = (gamma*dvy*vy/aV)[:,:,np.newaxis,np.newaxis]*(sinh_aV[:,:,np.newaxis,np.newaxis]*np.eye(2)+ ((cosh_aV - sinh_aV/aV)/aV)[:,:,np.newaxis,np.newaxis]*mV)  + (gamma*sinh_aV/aV)[:,:,np.newaxis,np.newaxis]*dmV
                dA = - self.Ah(dH)*dt
                tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
                for t in range(T-2):
                    tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs , dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
                dQ.append(tdQ/(dt*sigma))
            # wx
            dws = np.array([1,0])
            dA = self.Aw(dws)*dt
            tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
            for t in range(T-2):
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
            tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
            dQ.append(1/(dt*sigma)*tdQ)
            # wy
            dws = np.array([0,1])
            dA = self.Aw(dws)*dt
            tdQ = sparse.bmat([[sparse.csc_matrix((Ns,Ns)), - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-2)*Ns))]])
            for t in range(T-2):
                tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)), -dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA, - Qs@iDv@dA,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
            tdQ = sparse.bmat([[tdQ],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),- dA.T@iDv@Qs, dA.T@iDv@Qs@iDv@A + A.T@iDv@Qs@iDv@dA]])]])
            dQ.append(1/(dt*sigma)*tdQ)
            # log sigma 2
            tdQ2 = sparse.bmat([[Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-2)*Ns))]])
            for t in range(T-2):
                tdQ2 = sparse.bmat([[tdQ2],[sparse.bmat([[sparse.csc_matrix((Ns,(t)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A + Qs, -Qs@iDv@A,sparse.csc_matrix((Ns,(T-3-t)*Ns))]])]])
            tdQ2 = sparse.bmat([[tdQ2],[sparse.bmat([[sparse.csc_matrix((Ns,(T-2)*Ns)),-A.T@iDv@Qs, A.T@iDv@Qs@iDv@A]])]])
            dQ.append(-1/(dt*sigma)*tdQ2)
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
            Q_c_fac= self.cholesky(Q_c)
            if (Q_fac == -1) or (Q_c_fac == -1):
                return(self.like)
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            if self.r == 1:
                data = data.reshape(-1,1)
                mu_c = mu_c.reshape(-1,1)
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*np.log(tau)/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - tau/2*((data-self.S@mu_c)**2).sum()
            like =  -like/(self.S.shape[0]*self.r)
            return(like)
        
    def Aw(self,ws) -> sparse.csc_matrix:
        if self.Awnew is None:
            self.setClib()
        M, N, T = self.grid.shape
        obj = self.Awnew(M, N, ws, self.grid.hx, self.grid.hy)
        row = self.Awrow(obj)
        col = self.Awcol(obj)
        val = self.Awval(obj)
        self.Awdel(obj)
        rem = row != (M*N)
        row = row[rem]
        col = col[rem]
        val = val[rem]
        return(sparse.csc_matrix((val, (row, col)), shape=(M*N, M*N)))
    
    def Ah(self,Hs) -> sparse.csc_matrix:
        if self.AHnew is None:
            self.setClib()
        M, N, T = self.grid.shape
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
        if not os.path.exists(tmp + '/ccode/lib_Acw_2D_b%d.so'%(self.bc)):
            os.system('g++ -c -fPIC %s/ccode/Acw_2D_b%d.cpp -o %s/ccode/Acw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            import platform
            if  'Windows' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_Acw_2D_b%d.so -o %s/ccode/lib_Acw_2D_b%d.so %s/ccode/Acw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc)) # not tested
            elif  'Linux' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_Acw_2D_b%d.so -o %s/ccode/lib_Acw_2D_b%d.so %s/ccode/Acw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc))
            else:
                os.system('g++ -shared -o %s/ccode/lib_Acw_2D_b%d.so %s/ccode/Acw_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            os.system('rm %s/ccode/Acw_2D_b%d.o'%(tmp,self.bc))
        self.lib = ctypes.cdll.LoadLibrary('%s/ccode/lib_Acw_2D_b%d.so'%(tmp,self.bc))
        M, N, T = self.grid.shape
        self.Awnew = self.lib.Aw_new
        self.Awnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,shape = (2,)), ctypes.c_double,ctypes.c_double]
        self.Awnew.restype = ctypes.c_void_p
        self.Awrow = self.lib.Aw_Row
        self.Awrow.argtypes = [ctypes.c_void_p]
        self.Awrow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*5,))
        self.Awcol = self.lib.Aw_Col
        self.Awcol.argtypes = [ctypes.c_void_p]
        self.Awcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*5,))
        self.Awval = self.lib.Aw_Val
        self.Awval.argtypes = [ctypes.c_void_p]
        self.Awval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (M*N*5,))
        self.Awdel = self.lib.Aw_delete
        self.Awdel.argtypes = [ctypes.c_void_p]
        self.Awdel.restype = None
        
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
        self.lib = ctypes.cdll.LoadLibrary('%s/ccode/lib_AH_2D_b%d.so'%(tmp,self.bc))
        self.AHnew = self.lib.AH_new
        self.AHnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=4,shape = (M*N,4,2,2)), ctypes.c_double,ctypes.c_double]
        self.AHnew.restype = ctypes.c_void_p
        self.AHrow = self.lib.AH_Row
        self.AHrow.argtypes = [ctypes.c_void_p]
        self.AHrow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*9,))
        self.AHcol = self.lib.AH_Col
        self.AHcol.argtypes = [ctypes.c_void_p]
        self.AHcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*9,))
        self.AHval = self.lib.AH_Val
        self.AHval.argtypes = [ctypes.c_void_p]
        self.AHval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (M*N*9,))
        self.AHdel = self.lib.AH_delete
        self.AHdel.argtypes = [ctypes.c_void_p]
        self.AHdel.restype = None