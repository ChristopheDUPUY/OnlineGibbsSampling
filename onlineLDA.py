# -*- coding: utf-8 -*-
import pickle,numpy,time,sys,scipy
from numpy import log,exp,sqrt
from scipy.special import gamma,polygamma,psi,gammaln
##numpy.random.seed(100)

class OnlineEM_LDA:
    def __init__(self,V,batch_size,K,N,tau0,nb_GS=20,kappa=0.5,No_a=False,Gl='NR',Lo='NO'):
        self._K        = K
        self._N        = N
        self._V        = V
        self.nb_GS     = nb_GS
        
        self.No_A      = No_a
        self.Gl        = Gl
        self.Lo        = Lo
           
        # Initialization of parameters
           # eta1 = topic matrix (beta in LDA original paper)
        self.eta      = 0.001
        self.lambd    = self.eta +numpy.zeros((self._K,self._V))
        self.lamb_bar = self.lambd.copy()
        self.eta1   = self.lambd/self.lambd.sum(axis=1)[:,numpy.newaxis]

           # eta2 = prior on topic document distributions (alpha in LDA original paper)
        self.a_tilde = 1+numpy.ones(K)/K
        self.b_tilde = 0.5
        self.eta2  = (self.a_tilde-1)/self.b_tilde

        # Average of parameters
        self.eta1_bar = self.eta1.copy()
        self.eta2_bar = self.eta2.copy()
        eta = (self.eta1.copy(),self.eta2.copy())


        # Initialization of sufficient statistics (S1,S2)
        self.S1       = numpy.zeros((self._K,self._V))
        self.S2       = numpy.zeros(self._K)
        S = self.Compute_S_star(eta)
        self.S1 = self.lambd.copy(); self.S2 = S[1]
        self.S1_bar = self.S1.copy()
        self.S2_bar = self.S2.copy()

        # Parameters of the algorithm, the stepsize is self.gamma
        self.ite      = 1
        self.tau0     = tau0
        self.kappa    = kappa
        self.gamma    = (self.tau0)**(-self.kappa)


        self.perf = []
        self.perf_grad = []

        
        return;



############################
############################
############################
    def Key_Step(self,k_alg,batch_ids):
        """ Compute one step of the chosen algorithm. The inputs:

             - k_alg = Chosen algorithm, among {'G-OEM','G-OEM++','V-OEM','V-OEM++'}
             - batch_ids = list of words indices of the current minibatch,
        """
            
        if k_alg=='G-OEM':
            self.OnlineEM_step(batch_ids,GS=True,l_update=False)
        if k_alg=='G-OEM++':
            self.OnlineEM_step(batch_ids,GS=True,l_update=True)

        if k_alg=='V-OEM':
            self.OnlineEM_step(batch_ids,GS=False,l_update=False)
        if k_alg=='V-OEM++':
            self.OnlineEM_step(batch_ids,GS=False,l_update=True)

 

#####################################################################
    def OnlineEM_step(self,batch_ids,GS=True,l_update=False):
        """ Compute one step of online EM.
                          
             - batch_ids = list of words indices,
             - GS: if GS==True, "Gibbs sampling" is used to approximate current conditional; Otherwise, variational is used,
             - l_update: if l_update==True, local boosting is used (see the paper for details)
        """
        self.gamma = self.tau0*(self.ite+1)**(-self.kappa)

        eta   = (self.eta1.copy(),self.eta2.copy())
        S     = (self.S1.copy(),self.S2.copy())

        E_S_z   = self.Expected_S_z(batch_ids,S,eta,self.gamma,GS,l_update)

        self.S1 = self.S1 + self.gamma* (E_S_z[0] -self.S1)
        self.S2 = self.S2 + self.gamma* (E_S_z[1] -self.S2)

        eta_new = self.Compute_Eta_star((self.S1,self.S2))
        
        self.eta1     = eta_new[0].copy();self.eta2 = eta_new[1].copy();

        # Average of parameters
        self.eta1_bar = self.eta1/(self.ite+1) + self.ite*self.eta1_bar/(self.ite+1)
        self.eta2_bar = self.eta2/(self.ite+1) + self.ite*self.eta2_bar/(self.ite+1)
            
        self.ite += 1
        
        return;
#####################################################################


####################
### EXPECTATIONS ###
####################
    def Expected_S_z(self,batch_ids,S,eta,rho,GS,l_update):
        """ Compute the expected sufficient statistics. Inputs:

             - batch_ids = list of words indices,
             - S         = current sufficient statistics,
             - eta       = current parameters,
             - rho       = current stepsize (needed when l_update==True),
             - GS        : if GS==True, "Gibbs sampling" is used to approximate current conditional; Otherwise, variational is used,
             - l_update  : if l_update==True, local boosting is used (see the paper for details),
             - return (S1_res,S2_res) = the average expected sufficuent statistics over the minibatch batch_ids
        """

        eta1  = eta[0];eta2  = eta[1];        
        beta  = eta1
        try:
            beta_s  = beta/beta.sum(axis=0)[numpy.newaxis,:]
        except:
            beta    = beta+1e-6
            beta_s  = beta/beta.sum(axis=0)[numpy.newaxis,:]

        alpha   = eta2
        s_alpha = alpha.sum()

        if GS:
            res = GibbsSampling(self._K,batch_ids,beta,beta_s,alpha,s_alpha,S,rho,self.nb_GS,l_update,self.a_tilde,self.b_tilde,No_a=self.No_A,Lo=self.Lo)
        else:
            res = VariationalUpdates(self._K,batch_ids,beta,alpha,S,rho,self.nb_GS,l_update,self.a_tilde,self.b_tilde,No_a=self.No_A,Lo=self.Lo)
                   
        S1_res = numpy.zeros((self._K,self._V))
        S2_res = numpy.zeros(self._K)
        if not self.No_A:
            S2_res  += res[1].sum(axis=0)/float(len(batch_ids))
            if self.Gl=='Gam':
                self.a_tilde += res[2]
                self.b_tilde += res[3]
                self.eta2    = (self.a_tilde-1)/self.b_tilde

        for i_data in range(len(batch_ids)):
            S1_res[:,batch_ids[i_data]] += res[0][i_data].T/float(len(batch_ids))


        return (S1_res,S2_res)





###############################
######## UPDATES ##############
###############################

    def Compute_S_star(self,eta):
        """ Compute sufficient statistics S given the parameters eta.
        """
        eta1    = eta[0];eta2  = eta[1]
        S1      = eta1
        S2      = polygamma(0,eta2) -polygamma(0,eta2.sum())
        
        return (S1,S2)

    def Compute_Eta_star(self,S):
        """ Compute the parameters eta given the sufficient statistics S.
        """
        S1   = S[0];S2 = S[1];
        eta1 = numpy.zeros(S1.shape)
        eta2 = numpy.zeros(S2.shape)

        eta1 = S1/S1.sum(axis=1)[:,numpy.newaxis]
        if not self.No_A:
            if self.Gl=='NR':
                eta2 = self.NewtonRaphson(S2)
            elif self.Gl=='FP':
                eta2 = IterativeML(self.eta2,S2)
            elif self.Gl=='Gam':
                eta2 = self.eta2.copy()
        else:
            eta2 = self.eta2.copy()


        return (eta1,eta2)

    def NewtonRaphson(self,S):
        """ Update in alpha when using Newton-Raphson algorithm (see LDA original paper or online LDA (Hoffman et al., 2010) for details).
        """
        alpha     = self.eta2.copy()
        converged = False
        ite       = 1
        while not converged:
            g    = S - polygamma(0,alpha) + polygamma(0,alpha.sum())
            t    = 1.0/float(ite)
            ite += 1
            z     = polygamma(1,self.eta2.sum())
            h     = -polygamma(1,self.eta2)
            c     = (g/h).sum()/( (1/z) + (1/h).sum() )
            while (alpha - t*(g-c)/h).min()<=0:
                t = t*0.8
            lastalpha  = alpha.copy()
            alpha = 0.0001 + alpha - t*(g-c)/h
            converged  = max(abs(alpha-lastalpha))<1e-2 or ite>10
        return alpha;
                    

############################
###### EVALUATION ##########
############################
    def Perplexity_test(self,ids,eta1,eta2,R=10,lf='0'):
        """ Compute the perplexity of a set of documents with 'left-to-right' algorithm (Wallach et al., 2009). Inputs:
          - ids       = list of words indices of test set
          - eta1/eta2 = parameters
          - R         = number of loops
          - return the average perplexity per document.
        """
        res     = 0
        Nd_t    = 0
        beta    = eta1.copy()
        try:
            beta_s  = beta/beta.sum(axis=0)[numpy.newaxis,:]
        except:
            beta    = beta+1e-6
            beta_s  = beta/beta.sum(axis=0)[numpy.newaxis,:]
        alpha   = eta2.copy()
        s_alpha = alpha.sum()
        strt = time.time()
        for d in range(len(ids)):
            doc = ids[d]
            Nd   = len(doc)
            Nd_t+= len(doc)
            tot_time = 0
            zz   = numpy.zeros((Nd,R,self._K))
            Nt   = numpy.zeros((R,self._K))
            for n in range(Nd):
                strt2 = time.time()
                p_n = 0
                tot_time2 = 0
                for r in range(R):
                    strt3 = time.time()
                    for n_p in range(n):
                        Nt[r,:]     = Nt[r,:]-zz[n_p,r,:]
                        prob        = beta[:,doc[n_p]]*((Nt[r,:] + alpha)/(n-1+s_alpha))
                        prob        = prob/prob.sum()
                        zz[n_p,r,:] = numpy.random.multinomial(1,prob)
                        Nt[r,:]     = Nt[r,:] + zz[n_p,r,:]
                    p_n      += (beta[:,doc[n]]*((Nt[r,:] + alpha)/(n+s_alpha)) ).sum()
                    prob      = beta[:,doc[n]]*((Nt[r,:] + alpha)/(n+s_alpha))
                    prob      = prob/prob.sum()
                    zz[n,r,:] = numpy.random.multinomial(1,prob)
                    Nt[r,:]   = Nt[r,:] + zz[n,r,:]
                p_n  = p_n/R
                res += log(p_n)
            if (d+1)%100==0 or d==0:
                if lf!='0':
                    logfile=open(lf,'a')
                    logfile.write('After %s documents: '%(str(d+1))+str(round(-res/(d+1),2))+'\t ('+str(round(time.time()-strt,1))+' seconds)\n')
                    logfile.close()
                else:
                    print 'After %s documents: '%(str(d+1))+str(round(-res/(d+1),2))+'\t ('+str(round(time.time()-strt,1))+' seconds)'

                        
        return -res/float(len(ids));



def GibbsSampling(K,b_ids,beta,beta_s,alpha,s_alpha,S,rho,ite_max,l_update,a_tilde,b_tilde,No_a=False,Lo='NO'):
    """ Approximate the conditional p(z|x,eta) for the current document with Gibbs sampling. Inputs:
         - b_ids          = list of words indices for the current minibatch,
         - beta/beta_s    = current topic matrix/sum of topics,
         - alpha/alpha_s  = current parameters/sum of parameters,
         - S              = current sufficient statistics,
         - rho            = current stepsize (needed when l_update==True),
         - ite_max        = number of Gibbs sample per document
         - l_update       : if l_update==True, local boosting is used (see the paper for details),
         - a_tilde/b_tilde: parameters for local update when the updates of alpha are done with a Gamma prior 
(cf the paper and single pass LDA (Sato et al.,2010) for details)
         - No_a           : if No_a==True, the parameter alpha is not updated.
         - Lo             = type of local updates among {'FP','Gam'}; 'FP'= fixed point method, 'Gam'=Gamma prior.
################
     Outputs:
         - res[i_d][i,:]  = approximate value for the vector of probabilities p(z_i|x,beta,alpha)
         - psi_a[i_d,:]   = vector of approximate values of the quantity E_{Z|X,beta,alpha}[\Psi( [alpha(s)]_k) + \sum_n z_{nk} ] for the document i_d,
which is the first term of the expected sufficient statistics E_{Z,theta|X,beta,alpha}[S2(X,Z)]
         - a_jt/b_j       : update direction for a_tilde/b_tilde when the local updates of alpha are done with a Gamma prior
    """
    S1     = S[0]; S2 = S[1]
    counts = []
    psi_a  = numpy.zeros((len(b_ids),K))
    N_k    = []
    res    = []
    ## Initialization of the counts
    for i_d in range(len(b_ids)):
        counts.append(numpy.zeros((len(b_ids[i_d]),K)))
        for i_w in range(len(b_ids[i_d])):
            counts[i_d][i_w,:] = numpy.random.multinomial(1,beta_s[:,b_ids[i_d][i_w]])
        N_k.append(counts[i_d].sum(axis=0))
        res.append(numpy.zeros((len(b_ids[i_d]),K)))

    ## Sampling
    # Proportion of sample taken into account in the final approximation
    prop_take = 0.25
    if Lo=='HALF':
        prop_take = 0.5
    alpha_1 = alpha.copy()
    a_jt = numpy.zeros(K)
    for ite in range(ite_max):
        b_j=0
        for i_d in range(len(b_ids)):
            # counts[i_d] = (matrix of size nb of words of document i_d times K) matrix of topic assignment of the current document (represented by i_d). All its coefficients are in {0,1}.
            # N_k[i_d]    = (vector of size K) histogram of topic assignments of the document i_d of the minibatch
            # p(z_i|z_{-j},x,beta,alpha) is proportional to p_val
            inds = range(len(b_ids[i_d]))
            numpy.random.shuffle(inds)
            coeffs = beta[:,b_ids[i_d]]/(len(b_ids[i_d])-1 + s_alpha)
            for i_w in inds:
                N_k[i_d]          -= counts[i_d][i_w,:]
                p_val              = coeffs[:,i_w]*(N_k[i_d] +alpha)
                p_val_s            = p_val.sum()
                n_ci               = numpy.random.multinomial(1,p_val/p_val_s)                    
                counts[i_d][i_w,:] = n_ci
                N_k[i_d]          += n_ci
            if ite/float(ite_max) >= (1-prop_take):
                for i_w in range(len(b_ids[i_d])):
                    p_val              = coeffs[:,i_w]*(N_k[i_d]-counts[i_d][i_w,:] +alpha)
                    res[i_d][i_w,:]   += (p_val/p_val.sum())/(prop_take*ite_max)
            psi_a[i_d,:] += (psi(alpha_1 + N_k[i_d] ) - psi(s_alpha + len(b_ids[i_d]) ) )

            if not No_a and Lo == 'Gam':
                a_jt  += ((psi(alpha_1 + N_k[i_d] ) - psi(alpha_1))*alpha_1)/(len(b_ids)*(ite+1))
                b_j   += (psi(len(b_ids[i_d]) + s_alpha) - psi(s_alpha) )/len(b_ids)


        if l_update:
            beta = (1-rho)*S1.copy()
            for i_d in range(len(b_ids)):
                beta[:,b_ids[i_d]] = beta[:,b_ids[i_d]] + rho*res[i_d].T/float(len(b_ids))
            sumbeta = numpy.zeros(beta.shape)+beta.sum(axis=1)[:,numpy.newaxis]
            beta = beta/sumbeta

            if not No_a:
                alpha = alpha_1.copy()
                if Lo=='FP':
                    alpha   = IterativeML(alpha,(1-rho)*S2+ rho*psi_a.sum(axis=0)/float((ite+1)*len(b_ids)))
                elif Lo=='Gam':
                    alpha   = (a_tilde + a_jt - 1)/(b_tilde + b_j)

        
    return res,psi_a/float(ite_max),a_jt,b_j




def VariationalUpdates(K,b_ids,beta,alpha,S,rho,ite_max,l_update,a_tilde,b_tilde,No_a=False,Lo='NO'):
    """ Approximate the conditional p(z|x,eta) for the current document with variational inference. Inputs:
         - b_ids          = list of words indices for the current minibatch,
         - beta           = current topic matrix,
         - alpha          = current parameters,
         - S              = current sufficient statistics,
         - rho            = current stepsize (needed when l_update==True),
         - ite_max        = number of Gibbs sample per document
         - l_update       : if l_update==True, local boosting is used (see the paper for details),
         - a_tilde/b_tilde: parameters for local update when the updates of alpha are done with a Gamma prior 
(cf the paper and single pass LDA (Sato et al.,2010) for details)
         - No_a           : if No_a==True, the parameter alpha is not updated.
         - Lo             = type of local updates among {'FP','Gam'}; 'FP'= fixed point method, 'Gam'=Gamma prior.
#############
     Outputs:
         - phi[i_d][i,:]   = variational approximate values for the vector of probabilities p(z_i|x,beta,alpha)
         - (psi(gamma) - psi(gamma.sum(axis=1))[:,numpy.newaxis])[i_d,:] = vector of approximate values of the 
quantity E_{Z|X,beta,alpha}[\Psi( [alpha(s)]_k) + \sum_n z_{nk} ] for the document i_d,
which is the first term of the expected sufficient statistics E_{Z,theta|X,beta,alpha}[S2(X,Z)]
         - a_jt/b_j        : update direction for a_tilde/b_tilde when the local updates of alpha are done with a Gamma prior
    """

    S1      = S[0]; S2 = S[1]
    phi     = [numpy.random.dirichlet(numpy.ones(K),len(b_ids[i_d]))  for i_d in range(len(b_ids))]
    gamma   = numpy.ones((len(b_ids),K))/K
    alpha_1 = alpha.copy()
    for ite_l in range(ite_max):
        a_jt = numpy.zeros(K); b_j=0
        for i_d in range(len(b_ids)):
            gamma[i_d,:] = alpha + phi[i_d].sum(axis=0)
            temp   = psi(gamma[i_d,:] )  + log(beta[:,b_ids[i_d]]).T
            phi[i_d] = temp - temp.max(axis=1)[:,numpy.newaxis] - log(exp(temp - temp.max(axis=1)[:,numpy.newaxis]).sum(axis=1))[:,numpy.newaxis]
            phi[i_d] = exp(phi[i_d])

            if not No_a and Lo=='Gam':
                a_jt  += ( (psi(alpha_1 + phi[i_d].sum(axis=0) ) - psi(alpha_1))*alpha_1 )/len(b_ids)
                b_j   += ( psi(len(b_ids[i_d]) + alpha_1.sum()) - psi(alpha_1.sum())     )/len(b_ids)


        if l_update:
            beta = (1-rho)*S1.copy()
            a_jt = numpy.zeros(K); b_j=0
            for i_d in range(len(b_ids)):
                beta[:,b_ids[i_d]] = beta[:,b_ids[i_d]] + rho*phi[i_d].T/float(len(b_ids))
                if not No_a and Lo=='Gam':
                    a_jt  += ( (psi(alpha_1 + phi[i_d].sum(axis=0) ) - psi(alpha_1))*alpha_1 )/len(b_ids)
                    b_j   += ( psi(len(b_ids[i_d]) + alpha_1.sum()) - psi(alpha_1.sum())     )/len(b_ids)
            sumbeta = numpy.zeros(beta.shape)+beta.sum(axis=1)[:,numpy.newaxis]
            beta = beta/sumbeta
            

            if not No_a:
                if Lo=='FP':
                    alpha = alpha_1.copy()
                    alpha   = IterativeML(alpha,(1-rho)*S2+ rho*(psi(gamma) - psi(gamma.sum(axis=1))[:,numpy.newaxis]).sum(axis=0)/float(len(b_ids)))
                elif Lo=='Gam':
                    alpha   = (a_tilde + a_jt - 1)/(b_tilde + b_j)

    return phi,(psi(gamma) - psi(gamma.sum(axis=1))[:,numpy.newaxis]),a_jt,b_j




def IterativeML(alpha,S):
    """ Update of alpha when "fixed point method" ('FP') is used (cf the paper or Minka [2000] for details).
    """
    for ite in range(100):
        g         = S + polygamma(0,alpha.sum())
        lastalpha = alpha.copy()
        alpha     = InverseDigamma(g)
    return alpha;

def InverseDigamma(y):
    gam = -psi(1)
    x0  = numpy.zeros(len(y))
    x0[y>=-2.22] = exp(y[y>=-2.22]) +0.5
    x0[y<-2.22]  = -1/(y[y<-2.22]+gam)
    
    for i in range(5):
        x0 = x0 - (psi(x0)-y)/polygamma(1,x0)
    return x0





def ExtractTopics(K,vocab,eta1,eta2,pref,words_max=20):
    """ Write topics in the file pref.
    """
    f0 = open(pref,'w')
    inds_K = sorted(range(K),key=lambda s:eta2[s],reverse=True)
    for i_k in range(K):
        k = inds_K[i_k]
        weights = sorted(enumerate(eta1[k,:]),key=lambda s:s[1],reverse=True)
        f0.write('TOPIC - '+str(i_k)+'('+str(k) +') ($\\alpha_k=%s$)\n'%( str(eta2[k]/eta2.sum()) ))
        for w in range(words_max):
            f0.write( vocab[weights[w][0]].replace('\n','')+'\t'+str(round(weights[w][1],4))+'\n')
        f0.write('\n\n\n')

    f0.close()
    return;







