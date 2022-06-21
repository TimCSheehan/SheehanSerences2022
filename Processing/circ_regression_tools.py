# `

import numpy as np
from sklearn import linear_model

# for IEM style fit only
# import fit_IEM_v2 as IEM2

n_chan = 9
ang_s = 180
xx = np.linspace(0,179,180)
cosd = lambda x : np.cos( np.deg2rad(x) )
make_basis_function = lambda xx,mu : np.power( (cosd(xx-mu) ), ( n_chan - (n_chan % 2) ) )
pi=np.pi

def gen_basis_set(shift=1):
    ang = np.arange(ang_s)
    chan_center = np.linspace(ang_s/n_chan, ang_s, n_chan,dtype=int) - shift
    basis_set = np.zeros((ang_s,n_chan))
    for c in np.arange(n_chan):
        basis_set[:,c] = make_basis_function(xx,chan_center[c])
    chan_center_mask = np.zeros(ang_s)==1
    chan_center_mask[chan_center]=1
    return basis_set, chan_center_mask

basis_set,chan_centers= gen_basis_set()
center_deg = xx[chan_centers]
center_rad = center_deg/90*pi
psi = np.exp(1j*center_rad)

def gen_design_matrix(ang):
    assert all(ang<ang_s),'Invalid Angles!'
    n_trials = len(ang)
    design_matrix  = np.zeros((len(ang),ang_s))
    for i in range(n_trials):
        this_angle = ang[i]
        design_matrix[i,this_angle] = 1
    return design_matrix

def circ_mean(w,alpha=None):
    # w- should be samples x degrees
    # alpha - defaults to 180 samples +/- pi
    
    if not alpha:
        alpha = np.linspace(0,2*np.pi,181)
        alpha = alpha[:-1]
    t = w * np.exp(1j * alpha)
    r = np.sum(t, axis=1)
    mu = np.angle(r)
    conf = np.abs(r)
    est = mu/np.pi*90
    
    return est,conf

def circ_regr(Xtrain,Ytrain,Xtest,want_vec_len=False):
    # Xtrain - samples x nvox
    # Ytrain - samples [0,180]
    
    assert Xtrain.shape[0] == len(Ytrain), 'Not matching sizes'
    Ytrain_ang = Ytrain/180*pi*2
    Ytrain_mat = np.stack((np.sin(Ytrain_ang),np.cos(Ytrain_ang))).T
    regr = linear_model.LinearRegression().fit(Xtrain,Ytrain_mat)
    out = regr.predict(Xtest)
    ang_hat = np.arctan(out[:,0]/out[:,1])
    
    inds = np.where(out[:,1]<0)[0]
    ang_hat[inds] = ang_hat[inds]+pi
    ang_hat = np.mod(ang_hat,2*pi)*180/pi/2
    if want_vec_len:
        vec_len = np.sqrt(np.sum(np.power(out,2),1)) # should indicate confidence?
        return ang_hat,vec_len
    return ang_hat

def circ_regr_IEM(Xtrain,Ytrain,Xtest):
    # crude piece of code to perform regression on IEM style channels
    #
    # Xtrain - samples x nvox
    # Ytrain - samples [0,180]
    
    assert Xtrain.shape[0] == len(Ytrain), 'Not matching sizes'
    Ytrain_mat = gen_design_matrix(Ytrain)@basis_set
    regr = linear_model.LinearRegression().fit(Xtrain,Ytrain_mat)
    out = regr.predict(Xtest)
    return out

def circ_regr_IEM_cval(X,Y,nFold = 10,want_act=False):
    # X - samples x nvox
    # Y - samples [0,180]
    n = len(Y)
    assert X.shape[0] == n, 'Not matching sizes' 
    # divy up data by blocks
    
    if type(nFold) is not int:
        G = nFold
        groups = np.unique(G)
        nFold = len(groups)
        train_mat = np.ones((nFold,n))
        for i in range(nFold):
            train_mat[i,G==groups[i]] = 0 
    else:
        sz_block = np.floor(n/nFold)
        b_start = np.arange(0,n,sz_block,dtype=int)
        train_mat = np.ones((nFold,n))
        for i in range(nFold):
            if i==nFold-1: train_mat[i,b_start[i]:] = 0 # make sure we include all values
            else: train_mat[i,b_start[i]:b_start[i+1]] = 0 
            
    train_mat = train_mat==1
    Yhat = np.zeros((n,9))
    for i in range(nFold):
        train_ind = train_mat[i,:]
        Yhat[~train_ind,:] = circ_regr_IEM(X[train_ind,:],Y[train_ind],X[~train_ind,:])
        
    dec = np.angle(Yhat@psi)*90/pi
    dec[dec<0]+=180
    if want_act:
        return dec,ang_hat
    else:
        return dec

    return Yhat
#     # center fits
#     Y_mids = np.linspace(0,180,10)[:-1]
#     diffs,YhatShift = np.zeros((n,9)),np.zeros((n,9))
#     for i in range(9):
#         diffs[:,i] = Y-Y_mids[i]
#     grp = np.argmin(np.abs(diffs),1)
    
#     for i in range(n):
#         YhatShift[i,:] = np.concatenate([Yhat[i,grp[i]:],Yhat[i,:grp[i]]] )

#     YhatShift = np.concatenate([YhatShift[:,4:],YhatShift[:,:4]],1)
#     return YhatShift

def circ_corr_pval(x,yi,nComp):
    y = yi.copy()
    true = circ_corr_coef(x, y)
    shuf = np.zeros(nComp)
    for i in range(nComp):
        np.random.shuffle(y)
        shuf[i] = circ_corr_coef(x, y)
    return true,np.mean(true<shuf)

def circ_corr_coef(x, y):
    """ calculate correlation coefficient between two circular variables
    Using Fisher & Lee circular correlation formula (code from Ed Vul)
    x, y are both in radians [0,2pi]
    
    """
    if np.any(x>90): # assume both variable range from [0,180]
        x = x/90*pi
        y = y/90*pi
    if np.any(x<0) or np.any(x>2*pi) or np.any(y<0) or np.any(y>2*pi):
        raise ValueError('x and y values must be between 0-2pi')
    n = np.size(x);
    assert(np.size(y)==n)
    A = np.sum(np.cos(x)*np.cos(y));
    B = np.sum(np.sin(x)*np.sin(y));
    C = np.sum(np.cos(x)*np.sin(y));
    D = np.sum(np.sin(x)*np.cos(y));
    E = np.sum(np.cos(2*x));
    Fl = np.sum(np.sin(2*x));
    G = np.sum(np.cos(2*y));
    H = np.sum(np.sin(2*y));
    corr_coef = 4*(A*B-C*D) / np.sqrt((np.power(n,2) - np.power(E,2) - np.power(Fl,2))*(np.power(n,2) - np.power(G,2) - np.power(H,2)));
    return corr_coef


def circ_regr_cval(X,Y,nFold = 10,want_vec_len=False):
    # X - samples x nvox
    # Y - samples [0,180]
    n = len(Y)
    assert X.shape[0] == n, 'Not matching sizes' 
    # divy up data by blocks
   
    
    if type(nFold) is not int:
        G = nFold
        groups = np.unique(G)
        nFold = len(groups)
        train_mat = np.ones((nFold,n))
        for i in range(nFold):
            train_mat[i,G==groups[i]] = 0 
    else:
        sz_block = np.floor(n/nFold)
        b_start = np.arange(0,n,sz_block,dtype=int)
        train_mat = np.ones((nFold,n))
        for i in range(nFold):
            if i==nFold-1: train_mat[i,b_start[i]:] = 0 # make sure we include all values
            else: train_mat[i,b_start[i]:b_start[i+1]] = 0 
                
    train_mat = train_mat==1
    Yhat,Ylen = np.zeros(n),np.zeros(n)
    for i in range(nFold):
        train_ind = train_mat[i,:]
        if want_vec_len:
            Yhat[~train_ind],Ylen[~train_ind] = circ_regr(X[train_ind,:],Y[train_ind],X[~train_ind,:],1)
        else:
            Yhat[~train_ind] = circ_regr(X[train_ind,:],Y[train_ind],X[~train_ind,:])
    if want_vec_len:
        return Yhat,Ylen
    else:
        return Yhat

def circ_corr_cval(X,Y,nFold=10,doPerm=100):
    Yhat_true = circ_regr_cval(X,Y,nFold)
    true_cc = circ_corr_coef(Y,Yhat_true)
    if not doPerm:
        return true_cc
    
    Y_perm = Y.copy()
    perm_vals = np.zeros(doPerm)
    for i in range(doPerm):
        np.random.shuffle(Y_perm)
        Yhat_perm = circ_regr_cval(X,Y_perm,nFold)
        perm_vals[i] = circ_corr_coef(Y,Yhat_perm)
    p_val = np.mean(perm_vals>true_cc)
    return true_cc,p_val,perm_vals
