import numpy as np
from scipy import sparse

class Grid:
    def __init__(self) -> None:
        self.sdim = 2
        self.type = "gridST"
        self.meta = "Regular Mesh in 2D and time"
        A = 40
        B = 40
        Tdur = 10
        self.M = 30
        self.N = 30
        self.T = 10
        self.Ns = self.M*self.N
        self.n = self.Ns*self.T
        self.hx = A/self.M
        self.hy = B/self.N
        self.V = self.hx*self.hy
        self.dt = 10/self.T
        self.x = np.linspace(self.hx/2,A-self.hx/2,self.M)
        self.y = np.linspace(self.hy/2,B-self.hy/2,self.N)
        self.t = np.linspace(self.dt/2,Tdur-self.dt/2,self.T)
        self.isExtended = False
        self.Ne = 0
        self.setGrid()
        self.setDv()
        self.S = None
        self.cov = None
        self.inter = False
        self.n2eidx = None
        self.Ae = None
        self.scale = False
        
    def setS(self):
        idxs = np.arange(self.M*self.N*self.T)
        idxs2 = self.n2e(idxs)
        S = sparse.csc_matrix((self.M*self.N*self.T,np.prod(self.shape))).tolil()
        S[idxs,idxs2] = 1
        if self.cov is not None:
            if self.inter:
                if self.scale:
                    self.S = sparse.bmat([[S,np.stack([np.ones(self.cov.shape[0]),self.cov/self.cov.max()],axis = 1)]]).tocsc()
                else:
                    self.S = sparse.bmat([[S,np.stack([np.ones(self.cov.shape[0]),self.cov],axis = 1)]]).tocsc()
            else:
                self.S = sparse.bmat([[S,self.cov.reshape(-1,1)]]).tocsc()
        elif self.inter:
            self.S = sparse.bmat([[S,np.ones(self.M*self.N*self.T).reshape(-1,1)]]).tocsc()
        else:
            self.S = S.tocsc()

    def getS(self, idxs = None) -> sparse.csc_matrix:
        if self.S is None:
            self.setS()
        if idxs is None:
            return(self.S)
        return(self.S.tolil()[idxs,:].tocsc())
        
    def addCov(self,cov: np.ndarray, inter = True,scale = False) -> None:
        self.inter = inter
        self.scale = scale
        self.cov = cov
        self.setS()
    
    def addInt(self) -> None:
        self.inter = True
        self.setS()
    
    def idx2pos(self,idx):
        pass
        
    def getIdx(self,pos: np.ndarray):
        """getIdx find the index of a position in the grid

        Parameters
        ----------
        pos : np.ndarray
            position in the grid (idx X, idx Y, idx T)
        """
        return((pos[0]+self.Ne)+(pos[1]+self.Ne)*(self.M+2*self.Ne)+ pos[2]*self.Ns)
    
    def n2e(self,idx):
        if self.n2eidx is None:
            self.n2eidx = {}
            for t in range(self.T):
                for j in range(self.N):
                    for i in range(self.M):
                        self.n2eidx[i+j*self.M+t*self.M*self.N] = (i+self.Ne)+(j+self.Ne)*(self.M+2*self.Ne)+t*(self.M+2*self.Ne)*(self.N+2*self.Ne)
        if hasattr(idx,"__len__"):
            return(np.array([self.n2eidx[i] for i in idx]))                
        return(self.n2eidx[idx])

    
    @property
    def shape(self):
        return([self.M+2*self.Ne,self.N+2*self.Ne,self.T])

    def extend(self,extend = 1) -> None:
        self.xe = np.hstack([np.linspace(self.x[0]-extend*self.hx,self.x[0],extend+1)[:-1],
                            self.x,
                            np.linspace(self.x[-1],self.x[-1]+extend*self.hx,extend+1)[1:]])
        self.ye = np.hstack([np.linspace(self.y[0]-extend*self.hy,self.y[0],extend+1)[:-1],
                            self.y,
                            np.linspace(self.y[-1],self.y[-1]+extend*self.hy,extend+1)[1:]])
        self.sxe, self.sye = np.meshgrid(self.xe,self.ye)
        self.sxe = self.sxe.flatten()
        self.sye = self.sye.flatten()
        self.isExtended = True
        self.Ne = extend
        self.Ns = (self.M+self.Ne*2)*(self.N+self.Ne*2)
        self.n = self.Ns*self.T
        self.basisN()
        self.basisH()
            
    def setGrid(self, x = None, y = None, t = None, extend = None) -> None:
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        self.t = self.t if t is None else t
        self.M = self.x.shape[0]
        self.N = self.y.shape[0]
        self.T = self.t.shape[0]
        self.A = self.x.max()-self.x.min()
        self.B =  self.y.max()-self.y.min()
        self.Tdur = self.t.max()-self.t.min()
        self.hx = self.A/(self.M-1)
        self.hy = self.B/(self.N-1) 
        self.V = self.hx*self.hy
        self.dt = self.Tdur/(self.T-1) 
        sx, sy = np.meshgrid(self.x,self.y)
        self.sx = sx.flatten()
        self.sy = sy.flatten()
        self.iy, self.it, self.ix = np.meshgrid(np.arange(self.N),np.arange(self.T),np.arange(self.M))
        self.ix = self.ix.flatten()
        self.iy = self.iy.flatten()
        self.it = self.it.flatten()
        self.basisN()
        self.basisH()
        self.basisA()
        self.Ns = self.M*self.N
        self.n = self.Ns*self.T
        self.V = self.hx*self.hy
        if extend is not None:
            self.extend(extend = extend)
        self.setDv()
    
    def setDv(self):
        self.Dv = self.V*sparse.eye(self.Ns)
        self.iDv = sparse.eye(self.Ns)/self.V
            

    def basis(self,dx = 0 , dy = 0, d = 2) -> (np.ndarray,np.ndarray):
        if self.isExtended:
            tx = self.sxe+dx
            ty = self.sye+dy
        else:
            tx = self.sx+dx
            ty = self.sy+dy
        if ((dx != 0) or (dy != 0)):
            xmin = self.sx.min()-self.hx/2
            xmax = self.sx.max()+self.hx/2
            ymin = self.sy.min()-self.hy/2
            ymax = self.sy.max()+self.hy/2
        else: 
            xmin = self.sx.min()
            xmax = self.sx.max()
            ymin = self.sy.min()
            ymax = self.sy.max()
        kx = np.linspace(xmin - 2*(xmax-xmin)/3, xmax + 2*(xmax-xmin)/3,8)
        ky = np.linspace(ymin - 2*(ymax-ymin)/3, ymax + 2*(ymax-ymin)/3,8)
        Bx = list([np.stack([((tx >= kx[i])&(tx < kx[i+1]) | ((tx >= kx[i])&(tx <= kx[i+1])&(i==(kx.size-2))))*1.0 for i in range(kx.size-1)],axis=1)])
        By = list([np.stack([((ty >= ky[i])&(ty < ky[i+1]) | ((ty >= ky[i])&(ty <= ky[i+1])&(i==(ky.size-2))))*1.0 for i in range(ky.size-1)],axis=1)])
        for r in range(1,d+1):
            Bx.append(np.zeros((tx.shape[0],kx.size-r-1)))
            By.append(np.zeros((ty.shape[0],ky.size-r-1)))
            for i in range(kx.size-r-1):
                Bx[r][:,i] = (tx - kx[i])/(kx[i+r]-kx[i])*Bx[r-1][:,i] + (kx[i+r+1] - tx)/(kx[i+r+1]-kx[i+1])*Bx[r-1][:,i+1]
                By[r][:,i] = (ty - ky[i])/(ky[i+r]-ky[i])*By[r-1][:,i] + (ky[i+r+1] - ty)/(ky[i+r+1]-ky[i+1])*By[r-1][:,i+1]
        bx = np.stack([Bx[2][:,0] + Bx[2][:,1],Bx[2][:,2],Bx[2][:,3] + Bx[2][:,4]],axis = 1)
        by = np.stack([By[2][:,0] + By[2][:,1],By[2][:,2],By[2][:,3] + By[2][:,4]],axis = 1)
        return(bx,by)

    def basisN(self) -> None:
        bx, by = self.basis()
        if self.isExtended:
            bs = np.zeros((self.sxe.shape[0],3*3))
        else:
            bs = np.zeros((self.sx.shape[0],3*3))
        for i in range(3):
            for j in range(3):
                    bs[:,i*3+j] = bx[:,j]*by[:,i]
        self.bs = bs


    def basisH(self) -> None:
        if self.isExtended:
            bxA = np.zeros((self.sye.shape[0],4,3))
            byA = np.zeros((self.sye.shape[0],4,3))
            bs = np.zeros((self.sxe.shape[0],4,3*3))
        else:
            bxA = np.zeros((self.sx.shape[0],4,3))
            byA = np.zeros((self.sy.shape[0],4,3))
            bs = np.zeros((self.sx.shape[0],4,3*3))
        for i in range(4):
            if (i == 0):
                bx,by = self.basis(dx=-1/2*self.hx)
            elif (i == 1):
                bx,by = self.basis(dx=1/2*self.hx)
            elif (i == 2):
                bx,by = self.basis(dy=-1/2*self.hy)
            elif (i == 3):
                bx,by = self.basis(dy=1/2*self.hy)
            bxA[:,i,:] = bx
            byA[:,i,:] = by
        for i in range(3):
            for j in range(3):
                    bs[:,:,i*3+j] = bxA[:,:,j]*byA[:,:,i]
        self.bsH = bs
        
    def basisA(self) -> None:
        if self.isExtended:
            bxA = np.zeros((self.sye.shape[0],4,3))
            byA = np.zeros((self.sye.shape[0],4,3))
            bs = np.zeros((self.sxe.shape[0],4,3*3))
        else:
            bxA = np.zeros((self.sx.shape[0],4,3))
            byA = np.zeros((self.sy.shape[0],4,3))
            bs = np.zeros((self.sx.shape[0],4,3*3))
        for i in range(4):
            if (i == 2):
                bx,by = self.basis(dx=-1/2*self.hx)
            elif (i == 0):
                bx,by = self.basis(dx=1/2*self.hx)
            elif (i == 3):
                bx,by = self.basis(dy=-1/2*self.hy)
            elif (i == 1):
                bx,by = self.basis(dy=1/2*self.hy)
            bxA[:,i,:] = bx
            byA[:,i,:] = by
        for i in range(3):
            for j in range(3):
                    bs[:,:,i*3+j] = bxA[:,:,j]*byA[:,:,i]
        self.bsA = bs

    def evalB(self,par,bs = None, d=None) -> np.ndarray:
        if d is not None:
            par = np.zeros(par.shape)
            par[d] = 1
        if bs is None:
            bs = self.bs
        return(bs@par)

    def evalBH(self,par,bs = None, d = None) -> np.ndarray:
        if d is not None:
            par = np.zeros(par.shape)
            par[d] = 1
        if bs is None:
            bs = self.bsH
        return(bs@par)
    
    def evalAdv(self,par,bs = None, d = None) -> np.ndarray:
        if d is not None:
            par = np.zeros(par.shape)
            par[d] = 1
        if bs is None:
            bs = self.bsA
        res = np.stack([bs[:,0,:]@par[:9],bs[:,1,:]@par[9:],bs[:,2,:]@par[:9],bs[:,3,:]@par[9:]],axis = 1)
        res = self.advBound(res)
        return(res)
    
    def assimilate_adv(self,we,wn) -> np.ndarray:
        Aa = np.stack([np.eye(self.M*self.N)]*4,axis = 2)
        for i in range(self.M):
            for j in range(self.N):
                k = i+j*self.M
                if i+1 == self.M:
                    Aa[k,k,0] += 1
                else:
                    Aa[k,k+1,0] += 1
                if (i - 1 == -1):
                    Aa[k,k,2] += 1
                else:
                    Aa[k,k-1,2] += 1
                if j+1 == self.N:
                    Aa[k,k,1] += 1
                else:
                    Aa[k,k+self.M,1] += 1
                if j-1 == -1:
                    Aa[k,k,3] += 1
                else:
                    Aa[k,k-self.M,3] += 1
        ww = np.zeros((we.shape[0],4))
        ww[:,0] = Aa[:,:,0]@we/2
        ww[:,1] = Aa[:,:,1]@wn/2
        ww[:,2] = Aa[:,:,2]@we/2
        ww[:,3] = Aa[:,:,3]@wn/2
        if self.isExtended:
            Ae = np.zeros((self.Ns,self.M*self.N))
            for i in range(self.M+self.Ne*2):
                for j in range(self.N+self.Ne*2):
                    k = i+j*(self.M+2*self.Ne)
                    if i < self.Ne and j < self.Ne:
                        if i >= j:
                            Ae[k,0] = j/self.Ne
                        else:
                            Ae[k,0] = i/self.Ne
                        continue
                    if i >= self.M + self.Ne and j >= self.N + self.Ne:
                        if self.N - j <= self.M - i:
                            Ae[k,self.M*self.N-1] = (self.N+self.Ne*2-1-j)/self.Ne
                        else:
                            Ae[k,self.M*self.N-1] = (self.M+self.Ne*2-1-i)/self.Ne
                        continue
                    if i >= self.M + self.Ne and j < self.Ne:
                        if self.M+self.Ne*2-1-i <= j:
                            Ae[k,self.M-1] = (self.M+self.Ne*2-1-i)/self.Ne
                        else:
                            Ae[k,self.M-1] = j/self.Ne
                        continue
                    if i < self.Ne and j >= self.N + self.Ne:
                        if i <= self.N+self.Ne*2-1-j:
                            Ae[k,self.M*(self.N-1)] = i/self.Ne
                        else:
                            Ae[k,self.M*(self.N-1)] = (self.N+self.Ne*2-1-j)/self.Ne
                        continue
                    if i < self.Ne:
                        Ae[k,(j-self.Ne)*self.M] = i/self.Ne
                    elif i >= self.M + self.Ne:
                        Ae[k,self.M-1+(j-self.Ne)*self.M] = (self.M+self.Ne*2-1-i)/self.Ne
                    if j < self.Ne:
                        Ae[k,i-self.Ne] = j/self.Ne
                    elif j >= self.N + self.Ne:
                        Ae[k,i-self.Ne+(self.N-1)*self.M] = (self.N+self.Ne*2-1-j)/self.Ne
                    if i >= self.Ne and j >= self.Ne and i < self.M + self.Ne and j < self.N + self.Ne:
                        Ae[k,(i-self.Ne) + (j-self.Ne)*(self.M)] = 1
            return(Ae@ww)
        else:
            return(ww)
        
        
    def advBound(self,ww):
        if self.isExtended:
            if self.Ae is None:
                Ae = np.zeros((self.Ns,self.M*self.N))
                for i in range(self.M+self.Ne*2):
                    for j in range(self.N+self.Ne*2):
                        k = i+j*(self.M+2*self.Ne)
                        if i < self.Ne and j < self.Ne:
                            if i >= j:
                                Ae[k,0] = j/self.Ne
                            else:
                                Ae[k,0] = i/self.Ne
                            continue
                        if i >= self.M + self.Ne and j >= self.N + self.Ne:
                            if self.N - j <= self.M - i:
                                Ae[k,self.M*self.N-1] = (self.N+self.Ne*2-1-j)/self.Ne
                            else:
                                Ae[k,self.M*self.N-1] = (self.M+self.Ne*2-1-i)/self.Ne
                            continue
                        if i >= self.M + self.Ne and j < self.Ne:
                            if self.M+self.Ne*2-1-i <= j:
                                Ae[k,self.M-1] = (self.M+self.Ne*2-1-i)/self.Ne
                            else:
                                Ae[k,self.M-1] = j/self.Ne
                            continue
                        if i < self.Ne and j >= self.N + self.Ne:
                            if i <= self.N+self.Ne*2-1-j:
                                Ae[k,self.M*(self.N-1)] = i/self.Ne
                            else:
                                Ae[k,self.M*(self.N-1)] = (self.N+self.Ne*2-1-j)/self.Ne
                            continue
                        if i < self.Ne:
                            Ae[k,(j-self.Ne)*self.M] = i/self.Ne
                        elif i >= self.M + self.Ne:
                            Ae[k,self.M-1+(j-self.Ne)*self.M] = (self.M+self.Ne*2-1-i)/self.Ne
                        if j < self.Ne:
                            Ae[k,i-self.Ne] = j/self.Ne
                        elif j >= self.N + self.Ne:
                            Ae[k,i-self.Ne+(self.N-1)*self.M] = (self.N+self.Ne*2-1-j)/self.Ne
                        if i >= self.Ne and j >= self.Ne and i < self.M + self.Ne and j < self.N + self.Ne:
                            Ae[k,(i-self.Ne) + (j-self.Ne)*(self.M)] = 1
                self.Ae = Ae    
            return(self.Ae@ww)
        else:
            return(ww)
        
        
    def plot(self,value):
        pass