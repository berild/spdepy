from netCDF4 import Dataset
import numpy as np
import datetime

# add check if file exists

def get_sinmod_training():
    nc = Dataset("../src/spdepy/data/SINMOD_27_05_21.nc")
    xtmp = np.array([ 99, 149,   1])
    ytmp = np.array([24, 69,  1])
    ttmp = np.array([  0, 144,   1])
    T = 10
    dt = 10.0
    xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
    ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])
    tdom = np.arange(ttmp[0],ttmp[1],ttmp[2])
    x = np.array(nc['xc'][xdom])
    y = np.array(nc['yc'][ydom])
    t = np.arange(0,T*10.0,T)
    data = np.array(nc['salinity'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0])
    ffile = 'SINMOD_27_05_21.nc'
    time = [(datetime.datetime(int('20'+ ffile.split('_')[3].split('.')[0]),int(ffile.split('_')[2]),int(ffile.split('_')[1]),0) + datetime.timedelta(minutes=x)).strftime("%H:%M") for x in nc['time'][tdom]*24*60]
    wn = np.array(nc['v_north'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0]).mean(axis=1)
    we = np.array(nc['u_east'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0]).mean(axis=1)
    mu = data.mean(axis = 1)
    
    mut = []
    muB = np.hstack([mu]*T)
    for i in range(0,144-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
    mut = np.array(mut).T - muB[:,np.newaxis]
    mut = mut + np.random.normal(size = mut.shape)*np.sqrt(1/3)
    
    rho = np.sum((data[:,1:]-mu[:,np.newaxis])*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]),axis = 1)/np.sum((data[:,:(data.shape[1]-1)] - mu[:,np.newaxis])**2,axis = 1)
    muf = (data[:,1:]-mu[:,np.newaxis]) - rho[:,np.newaxis]*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]) + np.random.normal(0,1/np.sqrt(3),data.shape[0]*(data.shape[1]-1)).reshape(data.shape[0],data.shape[1]-1)
    #muf = (data[:,1:]- data[:,:(data.shape[1]-1)]) + np.random.normal(0,1/np.sqrt(3),data.shape[0]*(data.shape[1]-1)).reshape(data.shape[0],data.shape[1]-1)

    max_step = 200
    lr = np.linspace(0.2,0.0001,max_step)
    tag = "sinmod_training"
    return({'x':x,'y':y,'t':t,'time':time,'data':data,'muf':muf,'mut':mut,'mu':mu,'muB':muB,'wn':wn,'we':we,'T':T,'dt':dt,'tag':tag})

def get_sinmod_validation():
    xtmp = np.array([ 99, 149,   1])
    ytmp = np.array([24, 69,  1])
    ttmp = np.array([  0, 144,   1])
    T = 10
    dt = 10.0
    xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
    ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])
    tdom = np.arange(ttmp[0],ttmp[1],ttmp[2])
    
    ffile = 'SINMOD_28_05_21.nc'
    nc = Dataset("../src/spdepy/data/"+ffile)
    x = np.array(nc['xc'][xdom])
    y = np.array(nc['yc'][ydom])
    t = np.arange(0,T*10.0,T)
    data = np.array(nc['salinity'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0])
    
    mut = []
    for i in range(0,144-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
    
    ffile = 'SINMOD_08_09_22.nc'
    nc = Dataset("../src/spdepy/data/"+ffile)
    data = np.array(nc['salinity'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0])
    for i in range(0,144-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        
    ffile = 'SINMOD_11_05_22.nc'
    nc = Dataset("../src/spdepy/data/"+ffile)
    data = np.array(nc['salinity'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0])
    for i in range(0,144-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
    mut = np.array(mut).T
    mut = mut + np.random.normal(size = mut.shape)*np.sqrt(1/3)
    
    ffile = 'SINMOD_22_06_22.nc'
    nc = Dataset("../src/spdepy/data/"+ffile)
    data = np.array(nc['salinity'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0])
    for i in range(0,144-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
    mut = np.array(mut).T
    mut = mut + np.random.normal(size = mut.shape)*np.sqrt(1/3)

    tag = "sinmod_validation"
    return({'x':x,'y':y,'t':t,'data':mut,'tag':tag})

def get_sinmod_test():
    xtmp = np.array([ 99, 149,   1])
    ytmp = np.array([24, 69,  1])
    ttmp = np.array([  0, 144,   1])
    T = 10
    dt = 10.0
    xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
    ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])
    tdom = np.arange(ttmp[0],ttmp[1],ttmp[2])
    
    ffile = 'SINMOD_27_05_21.nc'
    nc = Dataset("../src/spdepy/data/"+ffile)
    x = np.array(nc['xc'][xdom])
    y = np.array(nc['yc'][ydom])
    t = np.arange(0,T*10.0,T)
    data = np.array(nc['salinity'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0])
    wn = np.array(nc['v_north'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0]).mean(axis=1)
    we = np.array(nc['u_east'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0]).mean(axis=1)
    
    mut = []
    for i in range(0,144-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
    mut = np.array(mut).T
    mut = mut + np.random.normal(size = mut.shape)*np.sqrt(1/3)

    tag = "sinmod_validation"
    return({'x':x,'y':y,'t':t,'data':mut,'wn':wn,'we':we,'tag':tag})