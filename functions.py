import pandas as pd
import numpy as np
import lmfit

def U_V_Terzaghi_analytical(time,Cv,H):
    Tv=time*Cv/((H)**2)
    return np.sqrt(4*Tv/np.pi)/((1+(4*Tv/np.pi)**2.8)**0.179)
def FDM_V(Cc,Cr,sigmap,Cv,e0,H,load,two_way_drainage=True,dimz=15,dimt=4000,targettime=None):
    dz = H/dimz
    Z=np.arange(0,H,dz) 
    dimz=len(Z)
    dt=0.2*dz**2/np.max(Cv)
    try: #if targettime is not None
        dimt =np.int(np.ceil(targettime/dt))
    except:
        pass
    u=np.zeros((dimz,dimt))
    time=np.zeros((dimt))
    U=np.zeros((dimz,dimt))
    Uavg=np.zeros((dimt))
    sigmav=np.zeros((dimt))
    sigveff=np.zeros((dimt))
    sigp=np.zeros((dimt))
    e = np.zeros((dimt))
    e[0]=e0
    sigp[0]=sigmap
    sigmav[0]=np.interp(0,load.index,pd.to_numeric(load['Load']))
    u[0,:]=0 # 0 at top
    u[-1,:]=0 # 0 at bottom
    u[1:,0]=sigmav[0]
    U[0,:]=1
    for t in range(1,dimt):
        time[t]=time[t-1]+dt
        sigmav[t] = np.interp(time[t],load.index,pd.to_numeric(load['Load']))
        dsigdt = sigmav[t]-sigmav[t-1]
        for z in range(1,dimz-1):
            u[z,t]=Cv*dt/((Z[z]-Z[z-1])**2)*(u[z-1,t-1]+u[z+1,t-1]-2*u[z,t-1])+u[z,t-1]+dsigdt
            U[z,t]=1-(u[z,t]/u[z,0])
        # Boundary condition
        z=dimz-1
        if two_way_drainage:
            u[z,t]=0;#Cv*dt/(Z[z]-Z[z-1])**2*(u[z-1,t-1]+u[z-1,t-1]-2*u[z,t-1])+u[z,t-1]
        else:
            u[z,t]=Cv*dt/(Z[z]-Z[z-1])**2*(u[z-1,t-1]+u[z-1,t-1]-2*u[z,t-1])+u[z,t-1]+dsigdt
    Uavg=(1-(np.average(u[:,:],axis=0)/np.average(u[:,0]))) #only valid for single loads
    sigveff=sigmav-np.average(u[:,:],axis=0)
    for t in range(1,dimt):
        e[t]=e[t-1]-(Cr*np.log10((sigveff[t])/(sigveff[t-1])))*(sigveff[t]<sigp[t-1])-(Cc*np.log10((sigveff[t])/(sigp[t-1])))*(sigveff[t]>sigp[t-1])
        sigp[t]=sigveff[t]*(sigveff[t]>sigp[t-1])+sigp[t-1]*(sigveff[t]<=sigp[t-1])

    return time,Uavg,u,e,sigveff
def h5store(filename, df, tst, **kwargs):
    store = pd.HDFStore(filename)
    store.put(tst, df)
    store.get_storer(tst).attrs.metadata = kwargs
    store.close()
def h5load(store,tst):
    data = store[tst]
    metadata = store.get_storer(tst).attrs.metadata
    return data, metadata
def save_file_to_h5(save,data_store,method,dimension='',remarks=''):
    metadata={}
    metadata['dimension']=dimension
    metadata['remarks']=remarks
    for col in save.columns:
        metadata[col] = save[col].iloc[-1]
    h5store(data_store,save,(dimension+'-'+method+'-'+remarks),**metadata)
class LoadSteps:
    def __init__(self):
        self.load_steps = [0]
        self.duration = [0]
        self.start_time = [0]
        self.end_time = [0]
        self.load = [0]
        self.loaddf = pd.DataFrame(columns=['Load'])
    def add_load_step(self, duration, load):
        self.load_steps.append(len(self.load_steps))
        self.duration.append(duration)
        self.start_time.append(self.end_time[-1])
        self.end_time.append(self.start_time[-1]+duration)
        self.load.append(load)
        self.loaddf.loc[self.start_time[-1]]=self.load[-1]
        self.loaddf.loc[self.end_time[-1]-0.1]=self.load[-1]
    def change_load_step(self,load_step,duration,load):
        self.duration[load_step]=duration
        self.load[load_step]=load
        self.end_time[load_step]=self.start_time[load_step]+duration
        self.loaddf = pd.DataFrame(columns=['Load'])
        for i in range(1,self.load_steps[-1]+1):
            self.start_time[i]=self.end_time[i-1]
            self.end_time[i]=self.start_time[i]+self.duration[i]
            self.loaddf.loc[self.start_time[i]]=self.load[i]
            self.loaddf.loc[self.end_time[i]-0.1]=self.load[i]
    def delete_load_step(self,load_step):
        self.load_steps.remove(load_step)
        self.duration.remove(self.duration[load_step])
        self.load.remove(self.load[load_step])
        for i in range(load_step,self.load_steps[-1]):
            self.load_steps[i]=self.load_steps[i-1]+1
            self.start_time[i]=self.end_time[i-1]
            self.end_time[i]=self.start_time[i]+self.duration[i]
        self.loaddf = pd.DataFrame(columns=['Load'])
        for i in range(1,self.load_steps[-1]+1):
            self.loaddf.loc[self.start_time[i]]=self.load[i]
            self.loaddf.loc[self.end_time[i]-0.1]=self.load[i]  
#def create_save_file(): #TO BE FIXED
#    global save_file
#    save_file={}
#    return save_file
def lstsq_FDM_V(param,xsample,ysample,time_start,time_end,load,method,e0,H,test_list=[],weights=None,log_scale=None,mode='absolute',plotting=False,lim_num=15):
    '''
    - log_scale: dictionary per parameter indicating True or False
    - weights: dictionary with fitting weights for each load step
    - mode: fitting to 'absolute' void ratio or 'relative' void ratio change per load step
    '''
    try: 
        log_scale[list(param)[0]]
    except:
        log_scale={p:True for p in param}
    try:
        weights[0]
    except:
        weights={p:1 for p in xsample}
    pm={}
    for p in list(param):
        if log_scale[p]:
            pm[p]=10**param[p].value
        else:
            pm[p]=param[p].value
        
    global save_file
    print([p+'='+str(pm[p]) for p in pm if param[p].vary])
    time,Uavg,u,e,sigveff = FDM_V(pm['Cc'],pm['Cr'],pm['sigmap'],pm['Cv'],e0,H,load.loaddf,targettime=time_end[list(time_end)[-1]])
    print('simulation completed')
    lstsqlist=[]
    if mode == 'relative':
        rel=1
    else:
        rel=0
    for L in xsample.keys():
        if (time>time_start[L]).any():
            time_new = time[(time>=time_start[L]) & (time<=time_end[L])] -time[(time>=time_start[L]) & (time<=time_end[L])] [0]
            e1 = e[(time>=time_start[L]) & (time<=time_end[L])]-e[(time>=time_start[L])][0]*rel
            ysample[L]=ysample[L]-ysample[L][0]*rel
            y=np.interp(xsample[L][:lim_num],time_new,e1)
            lstsqlist.append(weights[L]*np.sum(((y-ysample[L][:lim_num])/(ysample[L][:lim_num]-(ysample[L][:lim_num]-1)*rel))**2)) #in case of relative: avoid exploding errors due to limited to no void ratio changes (i.e. during relaxation)
        elif time[-1]<time_end[L]:
            lstsqlist.append(100)
        if plotting==True:
            plt.plot(xsample[L],ysample[L],'o-',label='measurement; stage '+str(L))
            plt.plot(xsample[L],y,label='model; stage '+str(L))
    lstsq=np.sum(np.array(lstsqlist))
    #save_file[method]=save_file[method].append(pd.DataFrame(np.array([param[p].value for p in param if param[p].vary]+[lstsq])[:,None].transpose(),columns=[p for p in param if param[p].vary]+['lstsq']),sort=False)
    p#rint(method+'  iteration:'+str(len(save_file[method]))+'  lstsq = '+str(lstsq))
    if plotting==True:
        plt.legend();plt.show()
    return lstsq
def optim_minmaxrand(param,lstsq,load,xsample,ysample,time_start,time_end,log_scale,mode='absolute',remark_1='',plotting=False,method_local='lbfgsb',remarks='',data_store=''):
    method = 'optim_minmaxrand'
    global save_file
    save_file={}
    save_file[method]=pd.DataFrame(columns=[p for p in param if param[p].vary])
    
    comb={}
    out={}
    varyparam=[]
    for p in list(param):
        if param[p].vary==True:
            varyparam=varyparam+[p]
            comb[p]=[param[p].min+0.01]*(2**len(varyparam)-1)+[param[p].max-0.01]*(2**len(varyparam)-1)
            for k in varyparam[:-1]:
                comb[k]=comb[k]*2
    #last one first. max corner:
    min1=10;min2=50; min3=100
    min1_index=0; min2_index=1

    for p in varyparam:
        param[p].value=comb[p][-1]
    out[0] = lmfit.minimize(lstsq, param, args=(xsample,ysample,time_start,time_end,load,method),kws={'log_scale':log_scale,'mode':mode,'plotting':plotting},method=method_local)
    i=1;save_file[method].loc[i+10]=0
    min1=out[0].residual
    out_final = out[0]
    for p in varyparam:
        param[p].value=comb[p][0]
    out[1] = lmfit.minimize(lstsq, param, args=(xsample,ysample,time_start,time_end,load,method),kws={'log_scale':log_scale,'mode':mode,'plotting':plotting},method=method_local)
    i=2; save_file[method].loc[i+10]=0
    if out[1].residual<min1:
            min2=min1
            min1=out[1].residual
            out_final = out[1]
            min2_index=min1_index
            min1_index=1
    elif out[1].residual<min2:
        min2=out[1].residual
        min2_index=1

    param_diff=[]
    for p in varyparam:
        param_diff=param_diff+[np.abs(out[min1_index].params[p]-out[min2_index].params[p])/np.abs(np.average([out[min1_index].params[p],out[min2_index].params[p]]))]

    while (((min2/min1)>1.5) or ((min3/min2)>1.5)) or (np.max(param_diff)>0.2):
        for p in varyparam:
            param[p].value=param[p].min+np.random.rand()*(param[p].max-param[p].min)
        out[i] = lmfit.minimize(lstsq, param, args=(xsample,ysample,time_start,time_end,load,method),kws={'log_scale':log_scale,'mode':mode,'plotting':plotting},method=method_local)
        if remark_1=='':
            save_file_to_h5(save_file[method],data_store,method,dimension=mode,remarks=remarks)
        else:
            save_file_to_h5(save_file[method],data_store,method,dimension=remark_1,remarks=remarks)
        if out[i].residual<min1:
            min3=min2
            min2=min1
            min1=out[i].residual
            out_final = out[i]
            min2_index=min1_index
            min1_index=i
        elif out[i].residual<min2:
            min3=min2
            min2=out[i].residual
            min2_index=i
        elif out[i].residual<min3:
            min3=out[i].residual
        param_diff=[]
        for p in varyparam:
            param_diff=param_diff+[np.abs(out[min1_index].params[p]-out[min2_index].params[p])/np.abs(np.average([out[min1_index].params[p],out[min2_index].params[p]]))]

        print(i)
        i+=1
        print(str(min1)+'; '+str(min2)+'; '+str(min3)); print(np.max(param_diff))
    
