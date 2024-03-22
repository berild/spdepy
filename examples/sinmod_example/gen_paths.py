import numpy as np
import spdepy as sp
from tqdm import tqdm

def isVisited(pt, path):
    return np.array([np.any(np.all(path == pt[:,i], axis = 1)) for i in range(pt.shape[1])])


def isLegal(pt,M, N):
    return np.array([pt[0,i] >= 0 and pt[0,i] < M and pt[1,i] >= 0 and pt[1,i] < N for i in range(pt.shape[1])])

    
def generate_points(num_int = 12):
    data = sp.datasets.get_sinmod_training()
    M, N, P = data['x'].size, data['y'].size, data['t'].size
    rng = np.random.default_rng()
    pos1 = np.array([rng.integers(0,M),rng.integers(0,N)])
    while True:
        dist1 = rng.choice(np.arange(-num_int,num_int+1))
        dist2 = rng.choice([-1,1])*int(np.sqrt(num_int**2 - dist1**2))
        pos2 = pos1 + np.array([dist1,dist2])
        if pos2[0] >= 0 and pos2[0] < M and pos2[1] >= 0 and pos2[1] < N:
            break 
    posPos = []
    posProbs = []
    for i in np.arange(-num_int,num_int+1):
        tmp = int(np.sqrt(num_int**2-np.abs(i)**2))
        for j in [-tmp,tmp]:
            tmpPos = pos2 + np.array([i,j])
            if tmpPos[0] >= 0 and tmpPos[0] < M and tmpPos[1] >= 0 and tmpPos[1] < N:
                posPos.append(tmpPos)
                posProbs.append((((tmpPos - pos1)**2).sum())**2)
    posProbs = posProbs/np.sum(posProbs)
    pos3 = rng.choice(posPos,p = posProbs)
    posPos = []
    posProbs = []
    for i in np.arange(-num_int,num_int+1):
        tmp = int(np.sqrt(num_int**2-np.abs(i)**2))
        for j in [-tmp,tmp]:
            tmpPos = pos3 + np.array([i,j])
            if tmpPos[0] >= 0 and tmpPos[0] < M and tmpPos[1] >= 0 and tmpPos[1] < N:
                posPos.append(tmpPos)
                posProbs.append((((tmpPos - pos1)**2).sum())**2 + (((tmpPos - pos2)**2).sum())**2) 
    posProbs = posProbs/np.sum(posProbs)
    return np.stack([pos1,pos2,pos3,rng.choice(posPos,p = posProbs)],axis = 0)

def findPath(points):
    path = []
    data = sp.datasets.get_sinmod_training()
    M, N, P = data['x'].size, data['y'].size, data['t'].size
    path.append(points[0])
    pos = points[0]
    rng = np.random.default_rng()
    dirs = np.array([[-1,0],[0,-1],[1,0],[0,1],[1,1],[1,-1],[-1,1],[-1,-1]]).T
    for i in range(1,points.shape[0]):
        while not (pos == points[i]).all():
            candidates = pos[:,np.newaxis] + dirs
            legal = isLegal(candidates,M,N)
            visited = isVisited(candidates,path)
            dist = ((candidates - points[i][:,np.newaxis])**2).sum(axis = 0)
            dist = (1- (dist - dist.min())/(dist.max() - dist.min()))**3
            if sum(legal*~visited) == 1:
                dist = legal*~visited*1
            elif sum(legal*~visited) == 0:
                dist = dist*legal
            else: 
                dist = dist*legal*~visited
            prob = dist/dist.sum()
            if np.isnan(prob).any():
                print(prob)
                print(dist)
                print(candidates)
                print(legal)
                print(visited)
            iN = rng.choice(np.arange(candidates.shape[1]),p = prob)
            pos = candidates[:,iN]
            path.append(pos) 
    return np.stack(path,axis = 0)

def generate_path():
    points = generate_points(20)
    path = findPath(points)
    data = sp.datasets.get_sinmod_training()
    M, N, P = data['x'].size, data['y'].size, data['t'].size
    x, y = data['x'][path[:,0]], data['y'][path[:,1]]
    dist = 0 
    speed = 0.5 # m/s
    time = 0
    t = []
    t.append(0)
    for i in range(1,x.size):
        dist = dist + np.sqrt((y[i]-y[i-1])**2 + (x[i]-x[i-1])**2)
        time = int(dist/speed/60//10)
        t.append(time)
    i = path[:,0]
    j = path[:,1]
    t = np.array(t)
    rm = t < 10
    i = i[rm]
    j = j[rm]
    t = t[rm]
    idx = t*N*M + j*M + i 
    return idx

def main():
    n = 200
    for i in tqdm(range(n)):
        np.save("data/paths/%03d"%i,generate_path())
        
if __name__ == "__main__":  
    main()