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
        return(beta_boot)
    
    
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
    def Estim_H0(self,k):
        list_y_sample=list()
        Beta_ES=list()
        res_ES=list()
        if self.method=="linear":
            X=np.delete(self.X,k,axis=1).copy()
            model=st.families.Gaussian()
            Reg = st.GLM(self.y,X,model)
            results = Reg.fit()
            Beta=results.params
            resid=results.resid_deviance
            vraise=0
        if self.method=="logistic":
            X=np.delete(self.X,k,axis=1).copy()
            model=st.families.Binomial()
            Reg = st.GLM(self.y,X,model)
            results = Reg.fit()
            Beta=results.params
            resid=results.resid_pearson
            vraise=results.llf
        return(Beta,resid,X,vraise)
    
    def Fisher(self,X,X_sub,Y,gamma,beta):
        n=X.shape[0]
        p=X.shape[1]-1
        norm_h0=np.linalg.norm(Y-np.dot(X_sub,gamma))**2
        norm_h1=np.linalg.norm(Y-np.dot(X,beta))**2      
        return( (n-p)*(norm_h0-norm_h1)/norm_h1 )
    
    def p_value(self,F_stat,F_obs):
        return( np.sum( [1 for b in range(len(F_stat)) if F_stat[b] > F_obs] )/len(F_stat) )
    
    def bootstrap_H0_ES(self,k):
        #ETAPE 1: estime les paramètres et erreurs sous H0
        gamma,Residus,X_H0,vrais = self.Estim_H0(k)
        #ETAPE 2: échantillonne aléatoirement les résidus
        test_ES=[]
        for b in range(self.B):
            ind=npr.randint(0,len(Residus),len(Residus))
            #ETAPE 3: calcul des Beta et variances avec une regression linéaire sur l'échantillon bootstrapé
            var_H0=(1/(len(X_H0)-X_H0.shape[1]-1))*np.sum(Residus**2)
            y_hat=np.zeros(len(X_H0))
            for i in ind:
                if self.method == "linear":
                    y_hat[i] = np.dot(X_H0[i,:],gamma) + var_H0*Residus[i]
                if self.method == "logistic":
                    pi_logreg = np.exp( np.dot(X_H0[i,:],gamma) )
                    cutoff = pi_logreg / ( 1 + pi_logreg ) + np.sqrt(pi_logreg*(1-pi_logreg))*Residus[i]
                    y_hat[i] = 1 if cutoff > 0.5 else 0
            if self.method == "linear":
                model_sample =st.GLM(y_hat,X_H0,st.families.Gaussian())
            if self.method == "logistic":
                model_sample =st.GLM(y_hat,X_H0,st.families.Binomial())
            res_sample = model_sample.fit()
            beta_sample = res_sample.params
            #ETAPE 4: sauvegarde dans une liste les beta's et erreurs estimés
            X_sub=X_H0[ ind,: ]
            X = self.X[ind,:]
            Y_sub=self.y[ind]
            if self.method == "linear":
                test_ES.append( self.Fisher(X,X_sub,Y_sub,beta_sample,self.beta) )
            if self.method == "logistic":
                test_ES.append( -2*( res_sample.llf - vrais ) )
        return test_ES
          
    def bootstrap_H0_CS(self,k):
        #ETAPE 1: estime les paramètres et erreurs sous H0
        gamma,Residus,X_H0,vrais = self.Estim_H0(k)
        #ETAPE 2: échantillonne aléatoirement les résidus
        test_ES=[]
        for b in range(self.B):
            ind=npr.randint(0,len(Residus),len(Residus))
            #ETAPE 3: calcul des Beta et variances avec une regression linéaire sur l'échantillon bootstrapé
            if self.method == "linear":
                model_sample =st.GLM(self.y[ind],X_H0[ind,:],st.families.Gaussian())
            if self.method == "logistic":
                model_sample =st.GLM(self.y[ind],X_H0[ind,:],st.families.Binomial())
            res_sample = model_sample.fit()
            beta_sample = res_sample.params
            #ETAPE 4: sauvegarde dans une liste les beta's et erreurs estimés
            X_sub=X_H0[ ind,: ]
            X = self.X[ind,:]
            Y_sub=self.y[ind]
            if self.method == "linear":
                test_ES.append( self.Fisher(X,X_sub,Y_sub,beta_sample,self.beta) )
            if self.method == "logistic":
                test_ES.append( -2*( res_sample.llf - vrais ) )
        return test_ES
        
        