#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:16:28 2018

@author: yumin
"""




#### read in data
#import updateParameters as uP
import os
import sys
from random import shuffle
import numpy as np
import pickle
import time
import utils
#import updateParameters2

startTime = time.time()

#### initialization for synthetic data
sigma1 = 1.0 #2
sigma2 = 0.005
ALPHA = 0.0001 # concentration factor for Dirichlet process
BETA = 0.01#1.0#1 # coefficient for the exponiential decay
GAMMA = 0.01 #1 # coefficient for the time ? may be a function of time
RHO = 8.0 #2.0 # (0,) coefficient for energy function, the largeer, the stronger the constraint
Nc = 100 # num of clusters, initialized to be 200, can increase or decrease
tau0 = 0.01
C0 = 1.0
p0 = 0.5
fold = 1
repeat = 1
maxIter = 50#100 # maximum number of iteration
printDisplay = True # False

MINUTES = 23.5*60 # save results after how many minutes
#### synthetic data
dataname = '6models_wholeUSA_synthetic3_21random11_36months6to1p_'+str(BETA)+'_'+str(GAMMA)+'_'+str(RHO)+'_random.pkl'
filepathname = '../data/synthetic/'+dataname

#### save file setting
#### synthetic
savedirectory = '../results/synthetic/'
savedirectory = savedirectory+str(BETA)+'_'+str(GAMMA)+'_'+str(RHO)+'_'+str(sigma1)+'_'+str(sigma2)+'_'+str(ALPHA)
resultpath = savedirectory+'/'+str(fold)+'/'
filehandle = open(filepathname,'rb')
data = pickle.load(filehandle, encoding="latin1")
X = data['Xtrain'] # [N, D]
Y = data['Ytrain'] # [N, 1]
#Xinfotrain = data['Xinfotrain'] # [lon, lat, ilon, jlat, month]
Xinfotrain = data['Xinfotrain'] # [ilon, jlat, month]
AdjIndtrain = data['AdjIndtrain'] # [N, Nn]
AdjHtrain = data['AdjHtrain'] # [N, Nn]
dumpN, Nn = AdjIndtrain.shape
N, D = X.shape # N is number of training data, D is dimension of data
Xtest = data['Xtest'] # [Ntest, D]
Ytest = data['Ytest'] # [Ntest, 1]
Xinfotest = data['Xinfotest'] # [Ntest,3], [ilon, jlat, month]
AdjHtest = data['AdjHtest'] # [Ntest,Nn], energy function between test point and its train neighbor
AdjIndtest = data['AdjIndtest'] # [Ntest,Nn], 2d index of train neighbor
Ntest, dummyD = Xtest.shape # Ntest is number of training data, D is dimension of data
Nmon = data['Nmon']#60#300#60 #672 # number of total month used (totally 672 months)
Nlon = data['Nlon']#33#62#10#17#15#75 # total points of longitude
Nlat = data['Nlat']#17#26#10#6#30 # total points of latitude
Ntotal = N + Ntest
#Strue = data['Strue'] # true labels
#Ntrue = data['Ntrue'] # true number of labels
del data # release cache
filehandle.close()
sigmak = sigma2

#S = initS(Y,N) # [N+1,1], cluster indicator
S = np.random.randint(1,Nc+1,(N+1,1))#.tolist() # [N+1,1], cluster indicator
kk = 1
while(kk<=Nc):
    if(np.count_nonzero(S==kk)==0):
        S[S>kk] -= 1
        kk -= 1
        Nc -= 1
    kk += 1    
S[N,0] = -10 # set S[N,1] to be different than any cluster index
#W = np.random.randn(Nc,D).tolist() # each row is wk, the weights for cluster k
#%% initialization for spike and slab
lamb = 1.0 # not update
nv0 = 0.0 # not update
#sigma = 1.0 # update later
Sigma = (sigmak*np.ones((Nc,))).tolist()
R = np.eye(D)
invR = np.eye(D) # not update
Gamma = []
for k in range(Nc):
    Gamma.append(np.ones((D,1)))
tau =  tau0*np.ones((D,))
C = C0*np.ones((D,1)) # [K,1] not update
p = p0*np.ones((D,1)) # not update

#### begin iteration
NcList = [] # number of clusters in each iteration
for nIter in range(maxIter):
    if(printDisplay==True):
        print("processing " + str(nIter) + "-th iteration\n")
        print("eclipsed time is " + str((time.time()-startTime)/60.0) + " minutes\n")
        print("there are " + str(Nc) + " clusters\n")
    NcList.append(Nc)

    #### update for W, gamma and sigma
    #W, Gamma, Sigma = updateParameters2.updateW(Nc,N,S,X,Y,D,Sigma,Gamma,C,tau,invR,R,p)    
    W = []
    #W = np.zeros((Nc,D))
    for k in range(1,Nc+1):   
        ll = 1*(S[:N]==k).reshape((N,)) # 0-1 1d array indicating S[i]==k if ll[i]==1
        Xk = X[ll>0,:] # [Nk,D] Nk is the number of datapoint belonging to cluster k
        Yk = Y[ll>0,:] # [Nk,1]

        sigma = Sigma[k-1] # 
        gamma = Gamma[k-1] #[D,1]
        
        ## update W
        a = np.power(C,gamma).reshape((D,)) # [D,]
        invD = np.diag(np.reciprocal(np.multiply(a,tau))) # [D,D]
        invA = sigma**(-1)*np.dot(Xk.T,Xk)+np.dot(invD,np.dot(invR,invD)) # [D,D]
        A = np.linalg.inv(invA)
        mu = sigma**(-1)*np.dot(A,np.dot(Xk.T,Yk)).ravel()
        wk = np.random.multivariate_normal(mu,A).reshape((D,1))
        W.append(wk.ravel())  # list, [Nc,D]
        
# =============================================================================
#         ## update sigma, why sigma is too small?
#         Nk,dummy = Yk.shape
#         shape = 0.5*(Nk+nv0)
#         scale = 0.5*((np.linalg.norm(Yk-np.dot(Xk,wk))**2)+nv0*lamb)
#         sigma = np.random.gamma(shape=shape,scale=scale)
#         sigma = sigma**(-1)
#         Sigma[k-1] = sigma
# =============================================================================
        
    ## update gamma
    for j in range(0,D):
        tempgamma = np.zeros((D,1))
        tempgamma[:,0] = gamma[:,0]           
        tempgamma[j,0] = 1
        a1 = np.power(C,tempgamma).reshape((D,))
        D1 = np.diag(np.multiply(a1,tau))        
        sigma11 = np.dot(D1,np.dot(R,D1))
        invsigma1 = np.linalg.inv(sigma11)
        exp1 = float(-0.5*np.dot(wk.T,np.dot(invsigma1,wk)))
        f1 = float(1.0/(np.sqrt(2*np.pi))*np.sqrt(np.linalg.det(invsigma1))*np.exp(exp1)*p[j,0])        
        tempgamma[j,0] = 0
        a2 = np.power(C,tempgamma).reshape((D,))
        D2 = np.diag(np.multiply(a2,tau))        
        sigma22 = np.dot(D2,np.dot(R,D2))
        invsigma2 = np.linalg.inv(sigma22)
        exp2 = float(-0.5*np.dot(wk.T,np.dot(invsigma2,wk)))
        f2 = float(1.0/(np.sqrt(2*np.pi))*np.sqrt(np.linalg.det(invsigma2))*np.exp(exp2)*(1-p[j,0]))            
        #gamma[j,0] = np.random.binomial(1,f1/(f1+f2))
        f21 = np.sqrt(np.linalg.det(sigma11)/np.linalg.det(sigma22))*np.exp(exp1-exp2)*(1-p[j,0])/p[j,0]           
        gamma[j,0] = np.random.binomial(1,1.0/(1.0+f21))
    Gamma[k-1] = gamma

#    if Nc == 1:
#        break

    if((time.time()-startTime)/60.0>MINUTES):# save result only close to the terminating time
        ctrainRMSE, trainrmse = utils.calTrainRMSE(N,X,Y,S,W,Nc)    
        S,W,Nc = utils.rmsmallclusters(Nlon,Nlat,S0=S,W0=W,Nc0=Nc,AdjIndtrain=AdjIndtrain)       
        testrmse, Stotal = utils.calTestRMSE3(S,W,AdjHtest,AdjIndtest,Ntest,Ntotal,Nc,Nn,N,Xtest,Ytest)
        NcList = np.asarray(NcList)
        #utils.saveResults(resultpath,fold,nIter,Nmon,dataname,S,W,BETA,GAMMA,RHO,sigma1,
        #                  sigma2,ALPHA,N,Nc,NcList,D,testrmse,trainrmse,repeat,Nlat,Nlon,
        #                  Xinfotrain,Xinfotest,Ntest,Stotal,startTime)
        exit()
        ##pass

    #S, W, Gamma, Sigma = updateParameters2.updateS(S,W,Gamma,Sigma,Y,X,Nc,N,Nn,invR,ALPHA,C,D,tau,sigmak,gamma,AdjHtrain,AdjIndtrain,printDisplay=True)
    #### update for S, may change W too
    ## calculate Gaussian distribution for later use
    WT = np.asarray(W).T  #[D,Nc] matrix
    #Ym = np.repeat(Y,Nc,axis=1) # extend to [N,Nc] matrix by repeating column
    Ym = np.dot(Y,np.ones((1,Nc))) # extend to [N,Nc] matrix by repeating column
    ## G is [N,Nc] matrix containing likelihood of point i for cluster k
    #L = 1.0/(np.sqrt(2*np.pi)*sigma2)*np.exp(-0.5/(sigma2**2)*(Ym - np.dot(X,WT))**2)    
    Sigma0 = np.asarray(Sigma).reshape((1,Nc)) #[1,Nc]
    invSigma0 = np.reciprocal(Sigma0) 
    invSigmam1 = np.dot(np.ones((N,1)),np.sqrt(invSigma0))
    invSigmam2 = np.dot(np.ones((N,1)),invSigma0) # extend to [N,Nc] matrix by repeating row
    exp = -0.5*(np.multiply(invSigmam2,(Ym - np.dot(X,WT))**2))
    npexp = np.exp(exp)
    L = 1.0/(np.sqrt(2*np.pi))*np.multiply(invSigmam1,npexp)    
   
    cAdd = [] # save index of new clusters
    nAdd = 0 # num of new clusters
    datalist = list(range(1,N+1))
    shuffle(datalist)
    #for i in range(1,N+1):
    iNum = 0
    for i in datalist:
        iNum += 1
        if(printDisplay==True and iNum%1000==0):
            print("processing " + str(iNum) + "-th data point\n")
            print("eclipsed time is " + str((time.time()-startTime)/60.0) + " minutes\n")            
        ik = S[i-1,0] # cluster indicator for i-th datapoint
        ni = np.count_nonzero(S==ik) # num of datapoint in the same cluster
        if(ni==1): # if i is the only datapoint in the cluster, delete the cluster 
            S[i-1,0] = -1 # set -1 to avoid counting    
            del W[ik-1] # delete the ik-th weight vector 
            del Sigma[ik-1] # delete the ik-th cluster sigma scalar
            del Gamma[ik-1] # delete the ik-th indicator vector
            L = np.delete(L,ik-1,1) # delete the ik-th likelihood vector 
            S[S>ik] -= 1 # all the >ik indicator minus 1
            Nc -= 1 # eliminate a cluster
            sigma2 = sigmak
        else:
            sigma2 = Sigma[ik-1]
    
        Q = np.zeros((Nc+1,1)) # save the probability of drawing cluster k for S[i]  
        xi = X[[i-1],:].T
        yi = Y[i-1,0]
        a = np.power(C,gamma).reshape((D,)) # [D,]
        invD = np.diag(np.reciprocal(np.multiply(a,tau)))
        #A = 1.0/(sigma2**2)*np.dot(xi,xi.T) + 1.0/(sigma1**2)*np.eye(D)
        invsigma0 = np.dot(invD,np.dot(invR,invD))
        invsigma3 = 1.0/(sigma2**2)*np.dot(xi,xi.T) + invsigma0
        w = yi/(sigma2**2)*np.linalg.solve(invsigma3,xi)
        sigma3 = np.linalg.inv(invsigma3)
        tempexp = np.exp(-0.5*(yi**2/(sigma2**2)-np.dot(w.T,np.dot(invsigma3,w))))
        Q[0,0] = ALPHA*1.0/(np.sqrt(2*np.pi)*sigma2)*np.sqrt(np.linalg.det(sigma3)*np.linalg.det(invsigma0))*tempexp
        ## for existing cluster (k==1,2,...)  
        vech = AdjHtrain[i-1,:] # [Nn,] 1d array, f(dij) = GAMMA*exp(-dij)
        adjind = AdjIndtrain[i-1,:]
        SSj = np.dot(S[adjind],np.ones((1,Nc)))# extend to [Nn,Nc] matrix by repeating column
        ## calculate the num of data points belong to cluster k
        tempk = np.arange(1,Nc+1).reshape((1,Nc))# [1,Nc]
        K = np.dot(np.ones((Nn,1)),tempk)# extend to [Nn,Nc] matrix by repeating row
        ## calculate the num of data points belong to cluster k
        nv, nk = np.unique(S[S>0],return_counts=True)
        M = 1*(SSj==K) # [Nn,Nc]  
   
        energy = np.exp(np.dot(vech,M)) # [Nc,] 1d array
        ## calculate the likelihood term
        lik = L[i-1,:]
        Q[1:,0] = np.multiply(nk,np.multiply(energy,lik)) # probability of drawing from cluster k
     
        ## draw a cluster
        Q = 1.0*Q / np.sum(Q) # normalize 
        S[i-1,0] = np.sum(np.cumsum(Q)<np.random.random_sample())       
        if(S[i-1,0]==0):# a new cluster
            nAdd += 1
            cAdd.append(i-1) # save the index of datapoint to be new clusters
   
    ## decrease or increase number of clusters    
    for kk in range(0,nAdd):
        Nc += 1
        S[cAdd[kk],0] = Nc
        Sigma.insert(cAdd[kk],sigmak)
        Gamma.insert(cAdd[kk],np.ones((D,1)))
 
##### final last update W so as to aglin with S
#### update for W
    W = []
    #W = np.zeros((Nc,D))
    for k in range(1,Nc+1):   
        ll = 1*(S[:N]==k).reshape((N,)) # 0-1 1d array indicating S[i]==k if ll[i]==1
        Xk = X[ll>0,:] # [Nk,D] Nk is the number of datapoint belonging to cluster k
        Yk = Y[ll>0,:] # [Nk,1]       
        sigma = Sigma[k-1] # 
        gamma = Gamma[k-1] #[D,1]
        
        ## update W
        a = np.power(C,gamma).reshape((D,)) # [D,]
        invD = np.diag(np.reciprocal(np.multiply(a,tau))) # [D,D]
        invA = sigma**(-1)*np.dot(Xk.T,Xk)+np.dot(invD,np.dot(invR,invD)) # [D,D]
        A = np.linalg.inv(invA)
        mu = sigma**(-1)*np.dot(A,np.dot(Xk.T,Yk)).ravel()
        wk = np.random.multivariate_normal(mu,A).reshape((D,1))
        W.append(wk.ravel())  # list, [Nc,D]
        
        ## update gamma
        for j in range(0,D):
            tempgamma = np.zeros((D,1))
            tempgamma[:,0] = gamma[:,0]
            
            tempgamma[j,0] = 1
            a1 = np.power(C,tempgamma).reshape((D,))
            D1 = np.diag(np.multiply(a1,tau))        
            sigma11 = np.dot(D1,np.dot(R,D1))
            invsigma1 = np.linalg.inv(sigma11)
            exp1 = float(-0.5*np.dot(wk.T,np.dot(invsigma1,wk)))
            f1 = float(1.0/(np.sqrt(2*np.pi))*np.sqrt(np.linalg.det(invsigma1))*np.exp(exp1)*p[j,0])
            
            
            tempgamma[j,0] = 0
            a2 = np.power(C,tempgamma).reshape((D,))
            D2 = np.diag(np.multiply(a2,tau))        
            sigma22 = np.dot(D2,np.dot(R,D2))
            invsigma2 = np.linalg.inv(sigma22)
            exp2 = float(-0.5*np.dot(wk.T,np.dot(invsigma2,wk)))
            f2 = float(1.0/(np.sqrt(2*np.pi))*np.sqrt(np.linalg.det(invsigma2))*np.exp(exp2)*(1-p[j,0]))            
            f21 = np.sqrt(np.linalg.det(sigma11)/np.linalg.det(sigma22))*np.exp(exp1-exp2)*(1-p[j,0])/p[j,0]
            
            gamma[j,0] = np.random.binomial(1,1.0/(1.0+f21))
        Gamma[k-1] = gamma


ctrainRMSE, trainrmse = utils.calTrainRMSE(N,X,Y,S,W,Nc)

S,W,Nc = utils.rmsmallclusters(Nlon,Nlat,S0=S,W0=W,Nc0=Nc,AdjIndtrain=AdjIndtrain)

testrmse, Stotal = utils.calTestRMSE3(S,W,AdjHtest,AdjIndtest,Ntest,Ntotal,Nc,Nn,N,Xtest,Ytest)
NcList = np.asarray(NcList)

#utils.saveResults(resultpath,fold,nIter,Nmon,dataname,S,W,BETA,GAMMA,RHO,sigma1,
#                sigma2,ALPHA,N,Nc,NcList,D,testrmse,trainrmse,repeat,Nlat,Nlon,
#                Xinfotrain,Xinfotest,Ntest,Stotal,startTime)

print('testrmse='+str(testrmse))
print('trainrmse='+str(trainrmse))

#### for synthetic data
#nmi = utils.cal_nmi(Stotal,Strue,Ntotal)
Smap = utils.cal_Smap(Xinfotrain,Xinfotest,Nlon,Nlat,Nmon,Stotal)
utils.plot_3d_fig(Nlon,Nlat,Nmon,Smap)


endTime = time.time()
print("total running time is " + str((endTime-startTime)/60.0) + " minutes")











