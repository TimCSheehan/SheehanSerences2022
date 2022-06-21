import numpy as np
import pandas as pd
from os import listdir

def deconvSubj(subj,roi_want,nTRModel = 30,want_stick_funs=0):
    # load data
    NTR = 392
    TR_SHIFT = -16
#     nTRModel = 40
    trialPerRun = 17
#     tim = '/mnt/neurocube/local/serenceslab/tsheehan/'
#     task = 'CD_Task'
    s='/'

#     sub_path = tim + task +s+subj+s
    sub_path = '../DataRaw/' + subj + '_'
    sessions = ['sess0','sess1','sess2','sess3']
#     sessions = np.sort(listdir(sub_path))
    
    DAT_ALL,INFO_ALL,TR_per_session = [],[],[]
    print('Loading Data...')
    for sess in sessions:
        if (sess[0]=='.') or (sess[:9]=='localizer'):
            continue
        #- go into each session and grab DAT_CD_TASK and INFO (if exists)
#         samp = sub_path + sess +'/SamplesSeg/'
        
        
        this_dat = pd.Series()
        try:
            info_path = sub_path + sess + '_INFO_CD_Task.csv'
            this_info = pd.read_csv(info_path,index_col=0)
            for roi in roi_want:
                dat_path = sub_path + sess +'_' + roi + '_DAT_CD_Task.npy'
                this_d = np.load(dat_path,allow_pickle=False)
                this_dat[roi] = this_d
#             dat_path = samp + 'DAT_CD_Task.pickle'
#             info_path = samp + 'INFO_CD_Task.pickle'
#             this_dat = pd.read_pickle(dat_path) # nTR x nVox
            
            TR_per_session.append(this_dat.V2.shape[0])
            DAT_ALL.append(this_dat)
            INFO_ALL.append(this_info)
        except:
            print('Failed, probably havent processed subj yet or they have weird structure eg. Localizer folder')
            break
    INFO_ALL = pd.concat(INFO_ALL,ignore_index=1)
    cumsumTR = np.cumsum(np.concatenate(([0],TR_per_session)))
    n_sess = len(DAT_ALL)
    Y= np.vstack([DAT_ALL[i]['V1'] for i in range(n_sess)])
    nTR,nVox = np.shape(Y)
    
    # get design matrix
    print('Generating Design Matrix...')
    n_block = len(INFO_ALL.block.unique())
    n_cols = nTRModel

    Xs = np.zeros((nTR,n_cols))
    Xp = np.zeros((nTR,n_cols))
    Xrun = np.zeros((nTR,len(INFO_ALL.block.unique())))
    
    for i,s in enumerate(INFO_ALL.block.unique()):
        Xrun[NTR*i:NTR*(i+1),i]=1

    stick_all = []
    for ind,block in enumerate(INFO_ALL.block.unique()):
        these_ev = INFO_ALL.block==block
        this_sess = int(INFO_ALL[these_ev].sess.values[0][-1])
        TR_add_sess = cumsumTR[this_sess]
        TR_add_block = ind*NTR
        stim_ind,probe_ind = [],[]
        block_ind = np.arange(TR_add_block,TR_add_block+NTR)
        inds_stim,inds_probe=[],[]
        for i in range(trialPerRun):
            inds_stim.append(np.round(INFO_ALL[these_ev&(INFO_ALL.trial==i+1)&(INFO_ALL.event=='stim')].
                                     TR.values[0]).astype(int)+TR_add_sess-TR_add_block+TR_SHIFT)
            inds_probe.append(np.round(INFO_ALL[these_ev&(INFO_ALL.trial==i+1)&(INFO_ALL.event=='line')].
                                     TR.values[0]).astype(int)+TR_add_sess-TR_add_block+TR_SHIFT)
        x_sess,x_probe = np.zeros(NTR),np.zeros(NTR)
        x_sess[inds_stim] = 1 
        x_probe[inds_probe] = 1
        stick_all.append((x_sess,x_probe))

        for t in range(nTRModel):
            Xs[block_ind,t] = np.concatenate((np.zeros(t),x_sess[:NTR-t]))
            Xp[block_ind,t] = np.concatenate((np.zeros(t),x_probe[:NTR-t]))
    X = np.hstack((Xs,Xp,Xrun)) # nTR x (nTRModel*2+nBlock)

    # y TRs x voxels 
    # x TRs x conditions x nTRmodel (<-- wrong)
    HRF_stim,HRF_probe = pd.Series(),pd.Series()
    
    print('Fitting!')
    for roi in roi_want:
        Y = np.vstack([DAT_ALL[i][roi] for i in range(n_sess)]) # nTR total x n Voxel
        h = np.linalg.solve(X.T@X,X.T)@Y
        HRF_stim[roi] = h[:nTRModel,:]
        HRF_probe[roi] = h[nTRModel:nTRModel*2,:]
    if want_stick_funs:
        return HRF_stim,HRF_probe,stick_all
    return HRF_stim,HRF_probe
        