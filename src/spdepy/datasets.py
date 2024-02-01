from netCDF4 import Dataset
import numpy as np
import datetime
import os
import pandas as pd
# add check if file exists

def get_sinmod_training():
    tmp = os.path.dirname(__file__)
    ffile = 'SINMOD_27_05_21.nc'
    nc = Dataset(tmp+"/data/" + ffile)
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
    time = [(datetime.datetime(int('20'+ ffile.split('_')[3].split('.')[0]),int(ffile.split('_')[2]),int(ffile.split('_')[1]),0) + datetime.timedelta(minutes=x)).strftime("%H:%M") for x in nc['time'][tdom]*24*60]
    wn = np.array(nc['v_north'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0]).mean(axis=1)
    we = np.array(nc['u_east'][tdom,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],tdom.shape[0]).mean(axis=1)
    mu = data.mean(axis = 1)
    rng = np.random.default_rng(seed=69)
    mut = []
    muB = np.hstack([mu]*T)
    for i in range(0,144-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
    mut = np.array(mut).T - muB[:,np.newaxis]
    mut = mut + rng.normal(size = mut.shape)*np.sqrt(1/3)
    
    rho = np.sum((data[:,1:]-mu[:,np.newaxis])*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]),axis = 1)/np.sum((data[:,:(data.shape[1]-1)] - mu[:,np.newaxis])**2,axis = 1)
    muf = (data[:,1:]-mu[:,np.newaxis]) - rho[:,np.newaxis]*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]) + rng.normal(0,1/np.sqrt(3),data.shape[0]*(data.shape[1]-1)).reshape(data.shape[0],data.shape[1]-1)
    #muf = (data[:,1:]- data[:,:(data.shape[1]-1)]) + np.random.normal(0,1/np.sqrt(3),data.shape[0]*(data.shape[1]-1)).reshape(data.shape[0],data.shape[1]-1)

    lr = np.hstack([np.linspace(0.2,0.1,200),np.linspace(0.1,0.05,200),np.linspace(0.05,0.01,200),np.linspace(0.01,0.00000001,200)])
    tag = "sinmod_training"
    return({'x':x,'y':y,'t':t,'time':time,'data':data,'muf':muf,'mut':mut,'mu':mu,'muB':muB,'wn':wn,'we':we,'T':T,'dt':dt,'tag':tag,'date':'27.05.2021', 'lr':lr})

def get_sinmod_validation():
    xtmp = np.array([ 99, 149,   1])
    ytmp = np.array([24, 69,  1])
    T = 10
    xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
    ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])

    ffile = 'SINMOD_28_05_21.nc'
    tmp = os.path.dirname(__file__)
    nc = Dataset(tmp+"/data/" + ffile)
    x = np.array(nc['xc'][xdom])
    y = np.array(nc['yc'][ydom])
    t = np.arange(0,T*10.0,T)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    mut = []
    mutS = []
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_29_05_21.nc'
    nc = Dataset(tmp+"/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_04_05_22.nc'
    nc = Dataset(tmp+"/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_10_05_22.nc'
    nc = Dataset(tmp+"/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_11_05_22.nc'
    nc = Dataset(tmp+ "/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_21_06_22.nc'
    nc = Dataset(tmp+"/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_22_06_22.nc'
    nc = Dataset(tmp+"/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_08_09_22.nc'
    nc = Dataset(tmp+"/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    ffile = 'SINMOD_09_09_22.nc'
    nc = Dataset(tmp+"/data/" + ffile)
    data = np.array(nc['salinity'][:,0,ydom,xdom]).swapaxes(0,2).swapaxes(0,1).reshape(x.shape[0]*y.shape[0],-1)
    for i in range(0,data.shape[1]-15,15):
        mut.append(data[:,i:(i+T)].T.reshape(-1))
        mutS.append(data[:,i:(i+T)])
        
    mut = np.array(mut).T
    mutS = np.stack(mutS,axis = 2)
    rng = np.random.default_rng(seed=1234)
    mut = mut + rng.normal(size = mut.shape)*np.sqrt(1/3)
    mutS = mutS + rng.normal(size = mutS.shape)*np.sqrt(1/3)
    iS = np.array([rng.choice(x.shape[0], 200, replace=True) for i in range(1)]).T
    jS = np.array([rng.choice(y.shape[0], 200, replace=True) for i in range(1)]).T
    tS = np.array([rng.choice(t.shape[0], 200, replace=True) for i in range(1)]).T
    idxS = iS + jS*x.shape[0] 
    idx = iS + jS*x.shape[0] + tS*x.shape[0]*y.shape[0]
    
    tag = "sinmod_validation"
    return({'dataS':mutS, 'data':mut,'idxS': idxS, 'idx': idx,'tS': tS,'tag':tag})

def get_sinmod_test():
    circ = 40075000

    xtmp = np.array([ 99, 149,   1])
    ytmp = np.array([24, 69,  1])
    ttmp = np.array([  0, 144,   1])
    T = 10
    xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
    ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])

    ffile = ['AUV2_08_09_22']#,'AUV2_08_09_22']
    tmpF = os.path.dirname(__file__)
    nc = Dataset(tmpF+"/data/" + "SINMOD_27_05_21.nc")

    x = np.array(nc['xc'][xdom])
    y = np.array(nc['yc'][ydom])
    t = np.arange(0,T*10.0,T)

    M = x.shape[0]
    N = y.shape[0]
    lon = np.array(nc['gridLons'][:,:])
    lat = np.array(nc['gridLats'][:,:])

    glat = lat[ytmp[0]:ytmp[1],xtmp[0]:xtmp[1]].transpose().flatten()
    glon = lon[ytmp[0]:ytmp[1],xtmp[0]:xtmp[1]].transpose().flatten()

    tfile = ffile[0]
    ttmp = np.array(tfile.split('_')[1:],dtype="int32")
    time = np.array([datetime.datetime(2000 + ttmp[2] ,ttmp[1],ttmp[0],0) + datetime.timedelta(minutes=x) for x in nc['time'][:]*24*60])
    time_emu = np.array([datetime.datetime.timestamp(x) for x in time])

    rlat = list()
    rlon = list()
    rz = list()
    rsal = list()
    timestamp = list()
    for tfile in ffile:
        eta_hat_df = pd.read_csv(tmpF +'/data/'+ tfile + "_EstimatedState.csv")
        salinity_df  = pd.read_csv(tmpF + "/data/" + tfile + "_Salinity.csv")
        on = salinity_df.columns[np.where(['timestamp' in x for x in salinity_df.columns])[0][0]]
        df = pd.merge_asof(eta_hat_df,salinity_df,on=on,direction="nearest")

        trlat = (eta_hat_df[" lat (rad)"] + eta_hat_df[" x (m)"]*np.pi*2.0/circ).to_numpy()
        trlon = (eta_hat_df[" lon (rad)"] + eta_hat_df[" y (m)"]*np.pi*2.0/(circ*np.cos(trlat))).to_numpy()

        trz = df[' depth (m)'].to_numpy()
        rm = (trz > 0.25)*(trz < 1.0)
        trlat = (trlat*180/np.pi)[rm]
        trlon = (trlon*180/np.pi)[rm]
        trsal = df[' value (psu)'].to_numpy()[rm]
        trz = trz[rm]
        timestamp.append(df[on].to_numpy()[rm])
        rlat.append(trlat)
        rlon.append(trlon)
        rz.append(trz)
        rsal.append(trsal)

    timestamp = np.hstack(timestamp)
    rlat = np.hstack(rlat)
    rlon = np.hstack(rlon)
    rz = np.hstack(rz)
    rsal = np.hstack(rsal)

    timeidx = np.zeros(timestamp.shape)
    idxs = np.zeros(rsal.size)
    for i in range(rsal.size):
        tmp = np.nanargmin(np.sqrt((rlat[i]-glat)**2 + (rlon[i]-glon)**2))
        tmpx= np.floor(tmp/N).astype("int32")
        tmpy= tmp - tmpx*N
        idxs[i] = tmpy*M  + tmpx
        timeidx[i] = np.nanargmin((time_emu-timestamp[i])**2)
    timeidx = timeidx.astype('int32')

    u_idx = list()
    u_tidx = list()
    u_data = list()
    u_var = list()
    u_timestamp = list()
    uniQ_i = np.unique(idxs)
    uniQ_ti = np.unique(timeidx)
    for t in range(uniQ_ti.size):
        for i in range(uniQ_i.size):
            tmp = np.where((idxs == uniQ_i[i])*(timeidx == uniQ_ti[t]))[0]
            if tmp.size > 4:
                u_idx.append(uniQ_i[i])
                u_tidx.append(uniQ_ti[t])
                u_data.append(rsal[tmp].mean())
                u_var.append(rsal[tmp].var())
                u_timestamp.append(timestamp[tmp].mean())
    rm = np.array(u_var,dtype = "float64") > 0.001
    u_idx = np.array(u_idx)[rm]
    u_tidx = np.array(u_tidx)[rm]
    u_data = np.array(u_data)[rm]
    u_var = np.array(u_var)[rm]
    u_timestamp = np.array(u_timestamp)[rm]
    data  = pd.DataFrame({'idx': np.array(u_idx, dtype = "int32"), 'tidx': np.array(u_tidx,dtype = 'int32'), 'data': np.array(u_data,dtype = "float64"), 'var': np.array(u_var,dtype = "float64"), 'timestamp': np.array(u_timestamp,dtype = "float64")})
    tag = "sinmod_test"
    return({'data': data,'tag':tag})
    