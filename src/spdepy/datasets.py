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
    ttmp = np.array([  0, 144,   1])
    T = 10
    dt = 10.0
    xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
    ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])
    tdom = np.arange(ttmp[0],ttmp[1],ttmp[2])

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
    xtmp = np.array([ 99, 149,   1])
    ytmp = np.array([24, 69,  1])
    ttmp = np.array([  0, 144,   1])
    T = 10
    dt = 10.0
    xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
    ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])
    tdom = np.arange(ttmp[0],ttmp[1],ttmp[2])

    ffile = ['AUV_08_09_22','AUV2_08_09_22']
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
    res = []
    for tfile in ffile:
        ttmp = np.array(tfile.split('_')[1:],dtype="int32")
        time = np.array([datetime.datetime(2000 + ttmp[2] ,ttmp[1],ttmp[0],0) + datetime.timedelta(minutes=x) for x in nc['time'][:]*24*60])
        eta_hat_df = pd.read_csv(tmpF +'/data/'+ tfile + "_EstimatedState.csv")
        salinity_df  = pd.read_csv(tmpF + "/data/" + tfile + "_Salinity.csv")
        on = salinity_df.columns[np.where(['timestamp' in x for x in salinity_df.columns])[0][0]]
        df = pd.merge_asof(eta_hat_df,salinity_df,on=on,direction="nearest")
        circ = 40075000
        R = 6371 * 10**3
        rlat = (eta_hat_df[" lat (rad)"] + eta_hat_df[" x (m)"]*np.pi*2.0/circ).to_numpy()
        rlon = (eta_hat_df[" lon (rad)"] + eta_hat_df[" y (m)"]*np.pi*2.0/(circ*np.cos(rlat))).to_numpy()
        idx = np.arange(rlat.shape[0])
        tidx = np.zeros(idx.shape) + 1
        rz = df[' depth (m)'].to_numpy()[idx]
        rm = (rz > 0.15)*(rz < 1.5)
        rlat = (rlat[idx]*180/np.pi)[rm]
        rlon = (rlon[idx]*180/np.pi)[rm]
        rsal = df[' value (psu)'].to_numpy()[idx][rm]
        rz = df[' depth (m)'].to_numpy()[idx][rm]
        timestamp = df[on].to_numpy()[idx][rm]
        idxs = np.zeros(rsal.size)
        for i in range(rsal.size):
            tmp = np.nanargmin(np.sqrt((rlat[i]-glat)**2 + (rlon[i]-glon)**2))
            tmpx= np.floor(tmp/N).astype("int32")
            tmpy= tmp - tmpx*N
            idxs[i] = tmpy*M  + tmpx
        si = 0
        u_idx = list()
        u_data = list()
        u_sd = list()
        u_time = list()
        u_fold = list()
        for i in range(1,idxs.size):
            if idxs[i-1] != idxs[i]:
                ei = i
                u_idx.append(idxs[si])
                u_data.append(rsal[si:ei].mean())
                u_sd.append(rsal[si:ei].std())
                u_time.append(timestamp[si:ei].mean())
                u_fold.append(tidx[si])
                si = i
        time_emu = np.array([datetime.datetime.timestamp(x) for x in time])
        time_data = np.array(u_time)
        timeidx = np.zeros(time_data.shape)
        for i in range(time_data.shape[0]):
            timeidx[i] = np.nanargmin((time_emu-time_data[i])**2)
        timeidx = timeidx.astype('int32')
        u_sd  = np.array(u_sd)
        u_sd[u_sd < 0.0001] = 1.0
        data  = pd.DataFrame({'idx': np.array(u_idx).astype("int32"), 'tidx': timeidx.astype('int32'), 'data': u_data, 'sd': u_sd, 'timestamp': u_time, 'time': time[timeidx]})
        tidxu = data['tidx'].unique()
        tmpidx = np.arange(data['tidx'].min(),data['tidx'].min()+10)
        while tmpidx[0] < tidxu.max():
            res.append([data[data['tidx'] == j] for j in tmpidx])
            tidxS = np.where(tidxu>tmpidx[-1])[0]
            if tidxS.size == 0:
                break
            tmpidx = np.arange(tidxu[tidxS.min()],tidxu[tidxS.min()]+10)
    data = [pd.concat([res[0][i],res[1][i]],ignore_index=True) for i in range(10)]
    tag = "sinmod_test"
    return({'data': data,'tag':tag})

    # all measurments, idxs, and datetime
    # 09.09.2022
    # 08.09.2022 
    # 11.05.2022
    # 10.05 2022
    # 27.05.2021
    