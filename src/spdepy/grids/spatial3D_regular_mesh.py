import numpy as np

class Grid:
    def __init__(self):
        self.sdim = 3
        self.type = "gridS"
        self.meta = "Regular Mesh in 3D"
        self.A = 40
        self.B = 40
        self.C = 40
        self.M = 30
        self.N = 30
        self.P = 30
        self.hx = self.A/self.M
        self.hy = self.B/self.N
        self.hz = self.C/self.P
        self.V = self.hx*self.hy*self.hz
        self.x = np.linspace(self.hx/2,self.A-self.hx/2,self.M)
        self.y = np.linspace(self.hy/2,self.B-self.hy/2,self.N)
        self.z = np.linspace(self.hz/2,self.C-self.hz/2,self.P)
        self.sx, self.sy, self.sz = np.meshgrid(self.x,self.y,self.z)
        self.sx = self.sx.flatten()
        self.sy = self.sy.flatten()
        self.sz = self.sz.flatten()
        self.bs = None
        self.bsH = None

    def setGrid(self, M = None, N = None, P = None, x = None, y = None, z = None):
        self.x = x
        self.y = y
        self.z = z
        self.M =  M
        self.N =  N 
        self.P = P 
        self.A = self.x.max()-self.x.min()
        self.B =  self.y.max()-self.y.min()
        self.C = self.z.max()-self.z.min()
        self.hx = self.A/(self.M-1)
        self.hy = self.B/(self.N-1) 
        self.hz = self.C/(self.P-1)
        self.V = self.hx*self.hy*self.hz
        sx, sy, sz = np.meshgrid(self.x,self.y,self.z)
        self.sx = sx.flatten()
        self.sy = sy.flatten()
        self.sz = sz.flatten()
        self.basisN()
        self.basisH()
            

    def basis(self,dx = 0 , dy = 0, dz = 0, d = 2):
        tx = self.sx+dx
        ty = self.sy+dy
        tz = self.sz+dz
        if ((dx != 0) or (dy != 0) or (dz != 0)):
            xmin = self.sx.min()-self.hx/2
            xmax = self.sx.max()+self.hx/2
            ymin = self.sy.min()-self.hy/2
            ymax = self.sy.max()+self.hy/2
            zmin = self.sz.min()-self.hz/2
            zmax = self.sz.max()+self.hz/2
        else: 
            xmin = self.sx.min()
            xmax = self.sx.max()
            ymin = self.sy.min()
            ymax = self.sy.max()
            zmin = self.sz.min()
            zmax = self.sz.max()
        kx = np.linspace(xmin - 2*(xmax-xmin)/3, xmax + 2*(xmax-xmin)/3,8)
        ky = np.linspace(ymin - 2*(ymax-ymin)/3, ymax + 2*(ymax-ymin)/3,8)
        kz = np.linspace(zmin - 2*(zmax-zmin)/3, zmax + 2*(zmax-zmin)/3,8)
        #kx = np.linspace(xmin - (xmax- xmin)/np.sqrt(16/d)*d,xmax+(xmax-xmin)/np.sqrt(16/d)*d,d+6)
        #ky = np.linspace(ymin - (ymax- ymin)/np.sqrt(16/d)*d,ymax+(ymax-ymin)/np.sqrt(16/d)*d,d+6)
        #kz = np.linspace(zmin - (zmax- zmin)/np.sqrt(16/d)*d,zmax+(zmax-zmin)/np.sqrt(16/d)*d,d+6)
        Bx = list([np.stack([((tx >= kx[i])&(tx < kx[i+1]) | ((tx >= kx[i])&(tx <= kx[i+1])&(i==(kx.size-2))))*1.0 for i in range(kx.size-1)],axis=1)])
        By = list([np.stack([((ty >= ky[i])&(ty < ky[i+1]) | ((ty >= ky[i])&(ty <= ky[i+1])&(i==(ky.size-2))))*1.0 for i in range(ky.size-1)],axis=1)])
        Bz = list([np.stack([((tz >= kz[i])&(tz < kz[i+1]) | ((tz >= kz[i])&(tz <= kz[i+1])&(i==(kz.size-2))))*1.0 for i in range(kz.size-1)],axis=1)])
        for r in range(1,d+1):
            Bx.append(np.zeros((tx.shape[0],kx.size-r-1)))
            By.append(np.zeros((ty.shape[0],ky.size-r-1)))
            Bz.append(np.zeros((tz.shape[0],kz.size-r-1)))
            for i in range(kx.size-r-1):
                Bx[r][:,i] = (tx - kx[i])/(kx[i+r]-kx[i])*Bx[r-1][:,i] + (kx[i+r+1] - tx)/(kx[i+r+1]-kx[i+1])*Bx[r-1][:,i+1]
                By[r][:,i] = (ty - ky[i])/(ky[i+r]-ky[i])*By[r-1][:,i] + (ky[i+r+1] - ty)/(ky[i+r+1]-ky[i+1])*By[r-1][:,i+1]
                Bz[r][:,i] = (tz - kz[i])/(kz[i+r]-kz[i])*Bz[r-1][:,i] + (kz[i+r+1] - tz)/(kz[i+r+1]-kz[i+1])*Bz[r-1][:,i+1]
        bx = np.stack([Bx[2][:,0] + Bx[2][:,1],Bx[2][:,2],Bx[2][:,3] + Bx[2][:,4]],axis = 1)
        by = np.stack([By[2][:,0] + By[2][:,1],By[2][:,2],By[2][:,3] + By[2][:,4]],axis = 1)
        bz = np.stack([Bz[2][:,0] + Bz[2][:,1],Bz[2][:,2],Bz[2][:,3] + Bz[2][:,4]],axis = 1)
        #bx = Bx[d]
        #by = By[d]
        #bz = Bz[d]
        #bx = np.delete(bx,1,1)
        #by = np.delete(by,1,1)
        #bz = np.delete(bz,1,1)
        #bx = np.delete(bx,Bx[d].shape[1]-2,1)
        #by = np.delete(by,By[d].shape[1]-2,1)
        #bz = np.delete(bz,Bz[d].shape[1]-2,1)
        #bx[:,0] = (Bx[d][:,0] + Bx[d][:,1])
        #by[:,0] = (By[d][:,0] + By[d][:,1])
        #bz[:,0] = (Bz[d][:,0] + Bz[d][:,1])
        #bx[:,Bx[d].shape[1]-3] = (Bx[d][:,Bx[d].shape[1]-1] + Bx[d][:,Bx[d].shape[1]-2])
        #by[:,By[d].shape[1]-3] = (By[d][:,By[d].shape[1]-1] + By[d][:,By[d].shape[1]-2])
        #bz[:,Bz[d].shape[1]-3] = (Bz[d][:,Bz[d].shape[1]-1] + Bz[d][:,Bz[d].shape[1]-2])
        return(bx,by,bz)

    def basisN(self):
        bx, by, bz = self.basis()
        bs = np.zeros((self.M*self.N*self.P,3*3*3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bs[:,i*3*3+j*3+k] = bx[:,j]*by[:,i]*bz[:,k]
        self.bs = bs


    def basisH(self):
        bxA = np.zeros((self.M*self.N*self.P,6,3))
        byA = np.zeros((self.M*self.N*self.P,6,3))
        bzA = np.zeros((self.M*self.N*self.P,6,3))
        for i in range(6):
            if (i == 0):
                bx,by,bz = self.basis(dx=-1/2*self.hx)
            elif (i == 1):
                bx,by,bz = self.basis(dx=1/2*self.hx)
            elif (i == 2):
                bx,by,bz = self.basis(dy=-1/2*self.hy)
            elif (i == 3):
                bx,by,bz = self.basis(dy=1/2*self.hy)
            elif (i == 4):
                bx,by,bz = self.basis(dz=-1/2*self.hz)
            elif (i == 5):
                bx,by,bz = self.basis(dz=1/2*self.hz)
            bxA[:,i,:] = bx
            byA[:,i,:] = by
            bzA[:,i,:] = bz
        bs = np.zeros((self.M*self.N*self.P,6,3*3*3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bs[:,:,i*3*3+j*3+k] = bxA[:,:,j]*byA[:,:,i]*bzA[:,:,k]
        self.bsH = bs

    def evalB(self,par,bs = None, d=None):
        if d is not None:
            par = np.zeros(par.shape)
            par[d] = 1
        if bs is None:
            bs = self.bs
        return(bs@par)

    def evalBH(self,par,bs = None, d = None):
        if d is not None:
            par = np.zeros(par.shape)
            par[d] = 1
        if bs is None:
            bs = self.bsH
        return(bs@par)
