import numpy as np
import numpy.random as npr
import pandas as pd
import math
import scipy as sc
from scipy.stats import norm
import statsmodels.genmod.generalized_linear_model as st

class Res():
    def __init__(self,X,y,method,alpha=0.05,B=1000):
        self.X=X  # !!! enlever y !!!
        self.y=y
        self.alpha=alpha
        self.B=B
        self.method=method
        if method=="linear":
            Reg = st.GLM(y,X)
        if method == "logistic":
            Reg = st.GLM(y,X,st.families.Binomial())
        results = Reg.fit()
        self.results=results
        #paramètres de sortie de la régression
        self.beta = results.params
        if method == "linear":
            self.resid = results.resid_deviance
        if method == "logistic":
            self.resid = results.resid_pearson
        self.var=(1/(len(X)-X.shape[1]-1))*sum(self.resid**2)
        self.y_pred = results.fittedvalues
        self.std_beta =  results.bse
        
    # Intervalle de confiance basique 
    def IC_base(self,beta_hat,beta):
        A= []
        beta=np.sort(np.array(beta),axis=1)
        for k in range(len(beta_hat)):
            B=[]
            B.append( 2*beta_hat[k]-beta[k,math.ceil(len(beta[0])*(1-self.alpha/2))-1] )
            B.append( 2*beta_hat[k]-beta[k,math.ceil(len(beta[0])*(self.alpha/2))-1] )        
            A.append(B)
        return(A)
    
    # Intervalle de confiance percentille
    def IC_perc(self,beta_hat,beta):
        A= []
        beta=np.sort(np.array(beta),axis=1)
        for k in range(len(beta_hat)):
            B=[]
            B.append(beta[k][math.ceil(len(beta[0])*(self.alpha/2))-1])
            B.append(beta[k][math.ceil(len(beta[0])*(1-self.alpha/2))-1])
            A.append(B)
        return(A)
    
    # Intervalle de confiance asymptotiquement normal
    def ICAN(self):
        beta = self.beta
        B=[]
        for d in range(len(beta)):
            A= []
            A.append(beta[d]-self.std_beta[d]/np.sqrt(len(self.X))*norm.ppf(1-self.alpha/2))
            A.append(beta[d]+self.std_beta[d]/np.sqrt(len(self.X))*norm.ppf(1-self.alpha/2))
            B.append(A)
        return B
    
    def case_sampling(self):
        beta_boot=[]
        sd_hat=np.zeros(self.B)
        for b in range(self.B):
            ind= npr.randint(0,len(self.X),len(self.X))
            sample_X=self.X[ind,:]
            sample_Y=self.y[ind]
            if self.method=="linear":
                model_sample = st.GLM(sample_Y,sample_X,st.families.Gaussian())
            if self.method=="logistic":
                model_sample = st.GLM(sample_Y,sample_X,st.families.Binomial())
            res_sample = model_sample.fit()
            beta_sample = res_sample.params
            beta_boot.append(beta_sample)
            sd_hat[b]=np.std(self.X[ind,:]/np.sqrt(np.var(self.X[ind,:])/np.mean(self.y[ind])**2+
                                    self.X[ind,:]*np.var(self.y[ind])/np.mean(self.y[ind])**4) )
        return(beta_boot,sd_hat)
    
    
    def errors_sampling(self):
        beta_boot=[]
        res_boot=[]
        for b in range(self.B):
            ind= npr.randint(0,len(self.X),len(self.X))
            sample_X=self.X[ind,:]
            sample_resid=self.resid[ind]
            sample_Y=np.zeros(len(self.X))
            if self.method=="linear":    
                for i in range(len(self.X)):
                    sample_Y[i] = np.dot(sample_X[i,:],self.beta) + np.sqrt(self.var)*sample_resid[i]
                model_sample =st.GLM(sample_Y,sample_X,st.families.Gaussian())
            if self.method=="logistic":
                for i in range(len(self.X)):
                    pi_logreg = np.exp( np.dot(sample_X[i,:],self.beta) )
                    cutoff = pi_logreg / ( 1 + pi_logreg ) + np.sqrt(pi_logreg*(1-pi_logreg))*sample_resid[i]
                    sample_Y[i] = 1 if cutoff > 0.5 else 0
                model_sample =st.GLM(sample_Y,sample_X,st.families.Binomial())
            res_sample = model_sample.fit()
            beta_sample = res_sample.params
            if self.method == "linear":
                resid_sample = res_sample.resid_deviance
            if self.method == "logistic":
                resid_sample = res_sample.resid_pearson
            sd_sample=np.std(resid_sample)
            beta_boot.append(beta_sample) 
            res_boot.append(sd_sample)
            
        return(beta_boot)
    
    # Cette fonction permet de réaliser une régression linéaire ou logistique sur l'un des deux datasets (variable choice)
    # En ayant préalablement retiré la colonne d'indice k (dans le but de faire un test de nullité)
    # Elle retourne les nouvelles estimations de Beta ainsi que les résidus associés
    # ETAPE 1
    def Estim_H0(self,k,hyp):
        y= self.y.copy()
        if hyp != 0:
            for i in range(len(y)):
                y[i]=y[i]-hyp*self.X[i,k]
        
        if self.method=="linear":
            X=np.delete(self.X,k,axis=1).copy()
            model=st.families.Gaussian()
            Reg = st.GLM(y,X,model)
            results = Reg.fit()
            Beta=results.params
            resid=results.resid_deviance
            vraise=0
        if self.method=="logistic":
            X=np.delete(self.X,k,axis=1).copy()
            model=st.families.Binomial()
            Reg = st.GLM(y,X,model)
            results = Reg.fit()
            Beta=results.params
            resid=results.resid_pearson
            vraise=results.llf
        return(Beta,resid,X,vraise,y)
    
    def Fisher(self,X,X_sub,Y,y_prime,gamma,beta):
        n=X.shape[0]
        p=X.shape[1]-1
        norm_h0=np.linalg.norm(y_prime-np.dot(X_sub,gamma))**2
        #print("erreur quadra H0",norm_h0)
        norm_h1=np.linalg.norm(Y-np.dot(X,beta))**2   
        #print("erreur quadra modele de base",norm_h1)
        return( (n-p)*(norm_h0-norm_h1)/norm_h1 )
    
    def p_value(self,F_stat,F_obs):
        return( np.sum( [1 for b in range(len(F_stat)) if F_stat[b] > F_obs] )/len(F_stat) )
    
    def p_value_boot(self,S_bar,S_b):
        card = len([b for b in range(self.B) if S_b[b]>S_bar])
        return ( card + 1 )/(self.B+1)
    
    def S_bar(self,k,hyp):
        sd_hat=np.std( self.X[:,k]/np.sqrt(np.var(self.X[:,k])/np.mean(self.y)**2+
                                    self.X[:,k]*np.var(self.y)/np.mean(self.y)**4) )
        return( abs( np.sqrt(len(self.X))*(self.beta[k]-hyp)/sd_hat ) )
    
    def bootstrap_H0_ES(self,k,hyp):
        #ETAPE 1: estime les paramètres et erreurs sous H0
        gamma,Residus,X_H0,vrais,y_prime = self.Estim_H0(k,hyp)
        var_H0=(1/(len(X_H0)-X_H0.shape[1]-1))*np.sum(Residus**2)
        #ETAPE 2: échantillonne aléatoirement les résidus
        test_ES=[]
        y_hat_list=[]
        ind_list=[]
        for b in range(self.B):
            ind=npr.randint(0,len(Residus),len(Residus))
            #ETAPE 3: calcul des Beta et variances avec une regression linéaire sur l'échantillon bootstrapé
            y_hat=np.zeros(len(X_H0))
            m=0
            for i in ind:
                if self.method == "linear":
                    y_hat[m] = np.dot(X_H0[i,:],gamma) + np.sqrt(var_H0)*Residus[i]
                if self.method == "logistic":
                    pi_logreg = np.exp( np.dot(X_H0[i,:],gamma) )
                    cutoff = pi_logreg / ( 1 + pi_logreg ) + np.sqrt(pi_logreg*(1-pi_logreg))*Residus[i]
                    y_hat[m] = 1 if cutoff > 0.5 else 0
                m=m+1
            y_hat_list.append(y_hat)
            ind_list.append(ind)
            if self.method == "linear":
                model_sample =st.GLM(y_hat,X_H0[ind,:],st.families.Gaussian())
            if self.method == "logistic":
                model_sample =st.GLM(y_hat,X_H0[ind,:],st.families.Binomial())
            res_sample = model_sample.fit()
            beta_sample = res_sample.params
            #ETAPE 4: sauvegarde dans une liste les beta's et erreurs estimés
            X_sub=X_H0.copy()#[ ind,: ]
            X = self.X.copy()#[ind,:]
            Y_sub=self.y.copy()#[ind]
            if self.method == "linear":
                test_ES.append( self.Fisher(X,X_sub,Y_sub,y_prime,beta_sample,self.beta) )
            if self.method == "logistic":
                test_ES.append( -2*( res_sample.llf - vrais ) )
        return test_ES, y_hat_list, ind_list
          
    def bootstrap_H0_CS_newmethod(self,k,hyp):
        S_b=np.zeros(self.B)
        #calcule gamma sous H0
        gamma = self.Estim_H0(k,hyp)[0]
        #calcule les beta bootstrapés
        beta_boot = self.case_sampling()[0]
        #calcule S_b
        sd_hat = self.case_sampling()[1]
        for b in range(self.B):
            S_b[b] = abs( np.sqrt(len(self.X))*(beta_boot[b][k] - self.beta[k])/sd_hat[b] )
        #calcule S_bar
        S_bar = self.S_bar(k,hyp)
        return(S_b,S_bar)
    
    def bootstrap_H0_CS(self,k,hyp):
        S_b = np.zeros(self.B)
        #ETAPE 1: estime les paramètres et erreurs sous H0
        gamma,Residus,X_H0,vrais,y_prime = self.Estim_H0(k,hyp)
        #ETAPE 2: échantillonne aléatoirement les résidus
        test_ES=[]
        for b in range(self.B):
            ind=npr.randint(0,len(Residus),len(Residus))
            #ETAPE 3: calcul des Beta et variances avec une regression linéaire sur l'échantillon bootstrapé
            if self.method == "linear":
                model_sample =st.GLM(y_prime[ind],X_H0[ind,:],st.families.Gaussian())
            if self.method == "logistic":
                model_sample =st.GLM(y_prime[ind],X_H0[ind,:],st.families.Binomial())
            res_sample = model_sample.fit()
            beta_sample = res_sample.params
            #ETAPE 4: sauvegarde dans une liste les beta's et erreurs estimés
            X_sub=X_H0.copy()#[ ind,: ]
            X = self.X.copy()#[ind,:]
            Y_sub=self.y.copy()#[ind]
            if self.method == "linear":
                test_ES.append( self.Fisher(X,X_sub,Y_sub,y_prime,beta_sample,self.beta) )
            if self.method == "logistic":
                test_ES.append( -2*( res_sample.llf - vrais ) )
        return test_ES

        
        