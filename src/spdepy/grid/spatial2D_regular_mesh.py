import numpy as np
from scipy import sparse

class Grid:
    def __init__(self) -> None:
        A = 40
        B = 40
        self.M = 30
        self.N = 30
        self.Ns = self.M*self.N
        self.n = self.Ns
        self.hx = A/self.M
        self.hy = B/self.N
        self.V = self.hx*self.hy
        self.x = np.linspace(self.hx/2,40-self.hx/2,self.M)
        self.y = np.linspace(self.hy/2,40-self.hy/2,self.N)
        self.isExtended = False
        self.Ne = 0
        self.setGrid()

    def getS(self, idxs = None) -> sparse.csc_matrix:
        if idxs is None:
            S = np.zeros((self.M*self.N,(self.M+self.Ne*2)*(self.N+self.Ne*2)))
            for i in range(self.M):
                for j in range(self.N):
                    ke = (i+self.Ne)+(j+self.Ne)*(self.M+2*self.Ne)
                    k = i+j*self.M
                    S[k,ke] = 1
            return(sparse.csc_matrix(S))
        else:
            return(sparse.csc_matrix((np.ones(idxs.shape[0]),(idxs[:,0],idxs[:,1])),shape=(self.M*self.N,(self.M+self.Ne*2)*(self.N+self.Ne*2))))
    
    def getIdx(self,pos: np.ndarray):
        """getIdx find the index of a position in the grid

        Parameters
        ----------
        pos : np.ndarray
            position in the grid (idx X, idx Y)
        """
        return((pos[0]+self.Ne)+(pos[1]+self.Ne)*(self.M+2*self.Ne))
    
    @property
    def shape(self):
        return([self.M+2*self.Ne,self.N+2*self.Ne])

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
        self.n = self.Ns
        self.basisN()
        self.basisH()
            
    def setGrid(self, x = None, y = None, extend = None) -> None:
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        self.M = self.x.shape[0]
        self.N = self.y.shape[0]
        self.A = self.x.max()-self.x.min()
        self.B =  self.y.max()-self.y.min()
        self.hx = self.A/(self.M-1)
        self.hy = self.B/(self.N-1) 
        self.V = self.hx*self.hy
        sx, sy = np.meshgrid(self.x,self.y)
        self.sx = sx.flatten()
        self.sy = sy.flatten()
        self.basisN()
        self.basisH()
        self.Ns = self.M*self.N
        self.n = self.Ns
        self.V = self.hx*self.hy
        if extend is not None:
            self.extend(extend = extend)
            

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
