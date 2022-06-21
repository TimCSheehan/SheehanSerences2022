import numpy as np
from minimize import run as minimize

# from fun_minLL_norm import fun_minLL_norm

from scipy.sparse import csr_matrix # compressed sparse row matrix
from scipy.linalg import lu # pivoted LU decomposition of matrix\

from scipy.integrate import quadrature as quad # quadgk
from scipy.optimize import fmin#,fmin_cg # fminsearch
# from scipy.optimize import 

class noiseModeling:
    """ Estimate orientation representations using Bayesian Inference and models of noise structure """
    def __init__(self,shut_up=0):
        
        self.nchans = 8 
        self.span = 180 
        self.n_init_noise_param =2 # 30
        self.tol = 1e-12
        self.n_probe_decode=180
        self.tuning_centers = np.arange(0,2*np.pi-.001,np.pi/4)
        self.psi = np.exp(1j*self.tuning_centers)
        self.shut_up = shut_up # verbosity
        
    def get_residuals(self,samples,ori):
        """ Get residules from linear encoding model
        
        Parameters
        ----------
        samples: array
            Training data (trials x voxels)
        ori: array
            Orientation presented in degrees [0,180] (trials)
        """
        assert (~np.any(np.isnan(samples))), 'Cannot have NaNs in input array'
        self.n_trials, self.n_vox = samples.shape
        self.fun_basis(ori/90*np.pi)
        self.W = np.linalg.pinv(self.YTrain)@samples # estimated weights, matches matlab mrdivide, aka '/'
        self.Winv = np.linalg.pinv(self.W)
        self.estimate = self.YTrain@self.W
        self.noise = samples - self.estimate  # residules of linReg (v + mu), TS 5/24/19: matches matlab
    
    def estimate_noise_parameters(self):
        """ estimate best parameters for noise covariance """
        
        # try recomended starting parameters before going random
        INIT = np.ones(self.n_vox+2)*0.7 # voxel noise
        INIT[-2]=0.3 # channel noise
        INIT[-1]=0.1 # global noise
        init = INIT.copy()

        self.solutions = np.zeros((self.n_vox+2,self.n_init_noise_param))
        self.fvals=np.ones(self.n_init_noise_param)*999

        for attempt in range(self.n_init_noise_param):
            try:
                self.solutions[:,attempt],fX,cc = minimize(self.fun_minLL_norm,init,length=1e8) # was 1e10 consider shortening
                self.fvals[attempt] = fX[-1]
            except:
                pass
            finally:
                init = INIT+np.random.randn(self.n_vox+2)*.1
                init[init<0]*=-1

        best_fval = np.min(self.fvals)
        assert (best_fval!=999), 'No good noise estimate found!'
        self.noise_params = self.solutions[:,np.argmin(self.fvals)]
        self.tau = self.noise_params[:self.n_vox] # voxel noise
        self.sig = self.noise_params[self.n_vox]  # channel noise
        self.rho = self.noise_params[self.n_vox+1] # global noise
        
        
    def decodeIEM(self,test_samples):
        assert (~np.any(np.isnan(test_samples))), 'Cannot have NaNs in input array'
        IEM = test_samples@self.Winv
        est = np.angle(IEM@self.psi)*90/np.pi+90
        unc = np.abs(IEM@self.psi)
        return self.foldPred(est),unc
    
    def foldPred(self,est_):
        est=est_.copy()
        est[est<90]+=180
        est-=90
        return est
        
    def decode(self,test_samples):
        """ Decode best estimate and uncertainty on each trial
        
        Parameters
        ----------
        test_samples: array
            Testing data (trials x voxels)
            
        Returns
        -------
        est: array
            single trial estimates (trials)
        unc: array
            single trial uncertainty (trials)
        posterior: array
            single trial likihood at all orientation (trials x 180)
        estInit: array
            single trial estimates without minimzation (trials)   
        """
        assert (~np.any(np.isnan(test_samples))), 'Cannot have NaNs in input array'
        
        n_trials_test = test_samples.shape[0]
        self.inits = np.linspace(0,2*np.pi,self.n_probe_decode)
        self.OMI = self.invSNC(0)
        est,unc  = np.zeros(n_trials_test),np.zeros(n_trials_test)
        estInit = np.zeros(n_trials_test)
        posterior = np.zeros((n_trials_test,self.n_probe_decode))
        
        for trial in range(n_trials_test):
            self.b = test_samples[trial,:]
            fvals = self.fun_LL(self.inits).ravel()
            self.minInitS=self.inits[np.argmin(fvals)]
            self.min_search()
            lik = self.fun_lik(self.inits)
            self.Integ,_ = quad(self.fun_lik,0,2*np.pi,tol=self.tol,maxiter=600) # numerically evalulate integral
            E1,_ = quad(self.fun_Eth1,0,2*np.pi,tol=self.tol,maxiter=600)
            est[trial] = np.mod(np.angle(E1),2*np.pi)/np.pi*90
            unc[trial] = np.sqrt(-2*np.log(np.abs(E1)))/np.pi*90
            posterior[trial,:] = lik
            estInit[trial] = self.minInitS/np.pi*90
            
#         est = self.foldPred(est)
#         estInit = self.foldPred(estInit)
        return est,unc,posterior,estInit

    def fit_CV_IEM(self,samples,ori,G):
        """ Perform cross-validated fits using grouping variable G
        
        Parameters
        ----------
        samples: array
            data (trials x voxels)
        ori: array
            Orientation presented in degrees [0,180] (trials)
        G: array
            Grouping varible (trials). 
            G==-1 will not be tested
            
        Returns
        -------
        dat: dict
            A structure with model outputs
        """
        
        n_trials_total = samples.shape[0]
        nGroup = len(np.unique(G))
        assert n_trials_total==len(G), 'G not correct length'
        est,unc = np.zeros(n_trials_total),np.zeros(n_trials_total)

        for g in np.unique(G):
            if g==-1:
                # set group not test with to -1
                print('skipping -1 group',end='')
                continue
            if ~self.shut_up: print(str(g) + ' ',end='')
            samples_train = samples[G!=g,:]
            ori_train = ori[G!=g]
            samples_test = samples[G==g,:]
            self.get_residuals(samples_train,ori_train)
            _est,_unc = self.decodeIEM(samples_test)
            est[G==g],unc[G==g]=_est,_unc
            
        est=est[G!=-1]
        unc=unc[G!=-1]
        dat = {'ori':ori[G!=-1],'estIEM':est,'uncIEM':unc}
        return dat
        
    
    def fit_CV(self,samples,ori,G):
        """ Perform cross-validated fits using grouping variable G
        
        Parameters
        ----------
        samples: array
            data (trials x voxels)
        ori: array
            Orientation presented in degrees [0,180] (trials)
        G: array
            Grouping varible (trials). 
            G==-1 will not be tested
            
        Returns
        -------
        dat: dict
            A structure with model outputs
        """
        
        n_trials_total = samples.shape[0]
        nGroup = len(np.unique(G))
        assert n_trials_total==len(G), 'G not correct length'
        est,unc,posterior = np.zeros(n_trials_total),np.zeros(n_trials_total),np.zeros((n_trials_total,self.n_probe_decode))
        est2,estIEM,uncIEM = np.zeros(n_trials_total),np.zeros(n_trials_total),np.zeros(n_trials_total)
        for g in np.unique(G):
            if g==-1:
                # set group not test with to -1
                print('skipping -1 group',end='')
                continue
             
            if ~self.shut_up: print(str(g) + ' ',end='')
            samples_train = samples[G!=g,:]
            ori_train = ori[G!=g]
            samples_test = samples[G==g,:]
            self.get_residuals(samples_train,ori_train)
            self.estimate_noise_parameters()
            _est,_unc,_post,_est2 = self.decode(samples_test)
            _estIEM,_uncIEM = self.decodeIEM(samples_test)
            est[G==g],unc[G==g],posterior[G==g,:]=_est,_unc,_post
            estIEM[G==g],uncIEM[G==g],est2[G==g]=_estIEM,_uncIEM,_est2
            
        est=est[G!=-1]
        unc=unc[G!=-1]
        posterior=posterior[G!=-1]
        est2=est2[G!=-1]
        estIEM=estIEM[G!=-1]
        uncIEM=uncIEM[G!=-1]
        dat = {'ori':ori[G!=-1],'estBNC':est,'estBNC_init':est2,'uncBNC':unc,'posterior':posterior,'fvals':self.fvals,
                         'solutions':self.solutions,'estIEM':estIEM,'uncIEM':uncIEM}
        return dat
            
    def fun_Eth1(self,s):
            out = (self.fun_lik(s)/self.Integ)*np.exp(1j*s)
            return out
    def fun_lik(self,s):
        ll = np.exp(-self.fun_LL(s)+self.mll) # just likelihood
        return np.array(np.hstack((ll)))
    def fun_LL(self,s):
        self.fun_basis(s)
#         bwc = np.tile(self.b,(len(s),1)).T - self.W@self.fun_basis(s).T
        bwc = np.tile(self.b,(len(s),1)) - (self.YTrain@self.W)
        negll = 0.5*self.MatProdDiag(bwc@self.OMI,bwc.T)
        return negll
    def MatProdDiag(self,M1,M2):
        M = np.multiply(M1,M2.T)
        out = np.sum(M,1)
        return out
    
    def min_search(self):
        self.mll = float(fmin(self.fun_LL,self.minInitS,maxiter=1e4,xtol=1e-5,full_output=True,disp=False)[1])
        # ,maxiter=1e4,xtol=1e-10
    def fun_minLL_norm(self,init): 
        """
        Return likelihood of observed residuals given noise parameters.
        init: array
            parameters (n_vox + 2)
        """
        # LL function for given noise covariance matrix
        # must pass variables as called by external general minimization function

        self.tau = init[:self.n_vox] # voxel noise
        self.sig = init[self.n_vox]  # channel noise
        self.rho = init[self.n_vox+1] # global noise
        # omi -> inverse of omega (see eq. 7)
        omi, NormConst = self.invSNC(want_ld=1) # inverse and logDet of noise cov-matrix       
        XXt = self.noise.T @ self.noise # empirical noise correlation
        negloglik = (1/self.n_trials) * self.MatProdTrace(XXt, omi) + NormConst # compute log-likelihood for this attempt
        if np.any(np.iscomplex(negloglik)): negloglik = np.inf #- degenerate solutions 
         
        # compute derivative
        der = np.full(init.shape[0], np.nan)
        JI = 1-np.eye(self.n_vox)
        R = np.eye(self.n_vox)*float(1-self.rho) + self.rho
        U = (omi @ self.noise.T) / np.sqrt(self.n_trials)
        dom = omi @ (np.eye(self.n_vox)-((1/self.n_trials) * XXt) @ omi)
        
        # TS: np.multiply critical! '*' gives matrix multiplication
        assert self.sig>0, 'neg variance!'
   
        der[:self.n_vox] = np.squeeze(2 * np.multiply(dom,R) @ self.tau)
        der[self.n_vox] = 2 * self.sig * self.MatProdTrace(self.W @ omi, self.W.T) - np.sum(np.sum(np.power((U.T * float(np.sqrt(2*self.sig)) @ self.W.T),2))) 
        der[self.n_vox+1] = np.sum(np.sum(np.multiply(dom,( np.multiply(np.outer(self.tau, self.tau),JI) ))) ) # TS need np.multiply      

        return float(negloglik), der

            
    def invSNC(self,want_ld=1):
        alpha = 1/(1-self.rho)
        Fi = alpha * csr_matrix(np.diag(np.power(self.tau,-2)), dtype=np.float64)
        ti = 1 / self.tau
        # TS 5/27/19 outer product! 
        Di = Fi - (float(self.rho*alpha**2) * np.outer(ti,ti) ) / (1 + (self.rho*self.n_vox*alpha))
        DiW = Di @ self.W.T
        WtDiW = self.W @ DiW
        # TS 5/23/19
        A = (1 / float(self.sig**2)*np.eye(self.nchans)+WtDiW) 
        omi = Di - ((np.linalg.solve(A.T @ A, A.T) @ DiW.T).T @ DiW.T) # verified TS 5/27
        # basically no recovery if get invalid solutions... just kill early! 
        assert self.rho<1, 'rho problem'
        assert (1 + self.rho*self.n_vox*alpha)>0 , 'other prob'
        assert ~np.any(self.tau<0), 'tau'
        if want_ld: 
            csr = csr_matrix(np.eye(self.nchans)) + float(self.sig**2) * WtDiW
            add_factors = np.log(1 + self.rho*self.n_vox*alpha) + self.n_vox*np.log(1-self.rho) + 2*np.sum(np.log(self.tau))  
            try:
                ld = self.my_logdet(csr, 'chol') + add_factors
            except np.linalg.LinAlgError:        
                print('Cholesky log determinant failed. Trying with LU decomposition.')
                ld = self.my_logdet(csr, 'lu') + add_factors
            except: 
                print("Unexpected error in invSNC")
            return omi, ld
        else:
            return omi

    def my_logdet(self,X, method):
        d = np.NaN
        if method == 'chol':
            d = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(X))), axis = 0)
        elif method == 'lu':
            P, L, U = lu(X)   #scipy.linalg.lu
            diag_upper = np.diag(U)
            c = np.linalg.det(P) * np.prod(np.sign(diag_upper))
            d = np.log(c) + np.sum(np.log(np.abs(diag_upper)), axis = 0)
        return d

            
    def MatProdTrace(self,A,B): return np.inner(A.flatten(),B.T.flatten())
        
    def fun_basis(self,ori):
        # ori - radians [0,2pi]
        self.YTrain = np.maximum(0,np.power(np.cos( np.tile(ori,(len(self.tuning_centers),1)).T -self.tuning_centers),5)) # differnet than np.max
