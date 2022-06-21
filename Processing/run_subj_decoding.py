import  numpy as np
import pandas as pd
import circ_corr as cc
import seaborn as sns
from os import path
pi=np.pi
import fit_IEM_v2 as IEM
import DecodeNoiseCorrelation as DNC

# data_root = '/mnt/neurocube/local/serenceslab/tsheehan/CD_Task/Analysis/SignalProcessingSeg/'
data_root = '../EpochedData/'
bad_sess = ['201910011846','201907231305'] # first one repeat from 54

def wrap(x):
    x[np.abs(x)>90]-=180*np.sign(x[np.abs(x)>90])
    return x


def run_subj(subj,rois,t_points,sav_root,just_IEM=0,paramsT={'donut':50,'anovaP':50}, paramsL={'donut':75,'anovaP':99}):
    D = DNC.noiseModeling()
    if just_IEM:
        CV_fun = D.fit_CV_IEM
    else:
        CV_fun = D.fit_CV
#     file = data_root + subj +'_dat_wTimeFix.pickle'
#     dat = pd.read_pickle(file)
    dat = pd.Series()
    dat['EV_TASK'] = pd.read_csv(data_root+subj+'_EV_TASK.csv',index_col=0)
    dat['EV_LOC'] = pd.read_csv(data_root+subj+'_EV_LOC.csv',index_col=0)
    d_task,d_loc = pd.Series(),pd.Series()
    for roi in rois:
        d_task[roi] = np.load(data_root+subj+'_DAT_TASK_'+roi+'.npy')
        d_loc[roi] = np.load(data_root+subj+'_DAT_LOC_'+roi+'.npy')
    dat['DAT_TASK'] = d_task
    dat['DAT_LOC'] = d_loc
    print('%s loaded' %subj)
    
    ev_loc = dat.EV_LOC.copy()
    donut_ev = ev_loc.d1.values==1
    ev_loc_train = ev_loc[donut_ev]
    ori_loc = D.foldPred(ev_loc_train.d0.values.copy())

    for roi in rois:
        print(roi)
        for t_use in t_points:
            settings = {'roi':roi,'t_use':t_use}
            sav_nameT = sav_root + subj +'_'+roi+'_T'+str(t_use)
            sav_nameL = sav_root + subj +'_'+roi
            sav_task,sav_locTask,sav_locLoc = sav_nameT+'_task.pickle', sav_nameT+'_locTask.pickle', sav_nameL+'_locLoc.pickle'
            if path.exists(sav_locTask):
                print('%s already exists, skipping!' %sav_locTask)
                continue
            else:
                print('running %s' %sav_locTask)

            dat_loc = dat['DAT_LOC'][roi] # jump right to these timepoints
            dat_task = np.nanmean(dat['DAT_TASK'][roi][:,:,t_use:t_use+4],2) # time x voxel x trials --> voxel x trial
            #### voxel & trial clean
            # Localizer
            vox_use,_ = voxel_selection(dat_loc,ev_loc,paramsT)
            _,vox_use_donut_only = voxel_selection(dat_loc,ev_loc,paramsL)
            dat_loc_train0 = dat_loc[vox_use,:]
            dat_loc_train1 = dat_loc[vox_use_donut_only,:]
            dat_loc_train = dat_loc_train0[:,donut_ev].T
            
            # Task
            ev_task_og = dat.EV_TASK.copy()
            bad_ev = np.isin(ev_task_og.sess,bad_sess)
            bad_trials = np.any(np.isnan(dat_task),0)
            print('Removing %d Trials with Nans' %np.sum(bad_trials))
            dat_task_train = dat_task[vox_use,:]
            dat_task_train = dat_task_train[:,~bad_ev&~bad_trials].T # get rid of NaNs
            
            ev_task = ev_task_og[~bad_ev&~bad_trials]
            ori_task = ev_task.orient0+90

            print('Task ',end='')
            # fit task LOBO
            sess_u=ev_task.sess.unique()
            n_sess = len(sess_u)
            G_task=np.ones(len(ev_task))
            for i,gi in enumerate(np.arange(0,n_sess,4)):
                G_task[np.isin(ev_task.sess,sess_u[gi:gi+4])] = i
            ev_task['G'] = G_task
            
            fit_task = pd.Series(CV_fun(dat_task_train,ori_task,G_task))
            fit_task['ev'] = ev_task
            fit_task['s'] = settings
            fit_task.to_pickle(sav_task)

            print('LocTask ',end='')
            # fit locTask
            dat_cat = np.vstack((dat_loc_train,dat_task_train))
            ori_cat = np.concatenate((ori_loc,ori_task))
            G_cat = np.ones(len(ori_cat))*-1
            G_cat[len(ori_loc):] = 1
            fit_locTask = pd.Series(CV_fun(dat_cat,ori_cat,G_cat))
            fit_locTask['ev'] = ev_task
            fit_locTask['Lev'] = ev_loc_train
            fit_locTask['s'] = settings
            fit_locTask.to_pickle(sav_locTask)
            
            print('Loc ',end='')
            
            # loc -> loc
            if path.exists(sav_locLoc):
                print('%s already exists, skipping!' %sav_locLoc)
                continue
            else:
                print('running %s' %sav_locLoc)
                
            ev_loc['d_ori'] = np.concatenate(([0],wrap(ev_loc.d0.values[:-1] - ev_loc.d0.values[1:])))
            donut_ev2 = (np.concatenate(([0],ev_loc.d1.values[:-1]))==1) & (ev_loc.d1.values==1)

            ev_loc_loc = ev_loc[donut_ev2]
            ori_loc_loc = D.foldPred(ev_loc_loc.d0.values.copy())
            dat_loc_loc = dat_loc_train1[:,donut_ev2].T
            G_loc = ev_loc_loc.sess.values
            fit_locLoc = pd.Series(CV_fun(dat_loc_loc,ori_loc_loc,G_loc))
            fit_locLoc['ev'] = ev_loc_loc
            fit_locLoc['s'] = settings
            fit_locLoc.to_pickle(sav_locLoc)    
    
def voxel_selection(samples,ev,params):
    # samples (voxels x trials)
    ind_all = np.arange(samples.shape[0])
    # donut selection
    donut_voxels = IEM.get_donut_mask(samples,ev,params['donut']) #minimun percentile
    donut_trials = (ev.d1.values==1)
    # anova selection
    samples_donut = samples[donut_voxels,:]
    samples_donut = samples_donut[:,donut_trials]
    ind_all_donut=ind_all[donut_voxels]
    
    ori = ev[donut_trials].d0.values
    ori_anova = np.round((ori-10)/2,-1)
    f_stats= IEM.anova1(samples_donut,ori_anova)
    n_keep = int(np.sum(donut_voxels)/2)
    ind_keep = np.argsort(f_stats)[-n_keep:]
    ind_all_both = ind_all_donut[ind_keep]
    print('%d -> %d voxels' %(samples.shape[0],len(ind_all_both)))
    return ind_all_both, ind_all_donut
    