# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:36:06 2019

@author: liuyuming
"""
import numpy as np
def calTrainRMSE(N,X,Y,S,W,Nc):
    trainRMSE = -1*np.ones((N,1))
    for ii in range(N):
        xii = X[ii,:].T
        yii = Y[ii,0]
        if(S[ii,0]-1>=len(W)): # skip deleted cluster
            continue
        wii = np.asarray(W[S[ii,0]-1])
        trainRMSE[ii,0] = (yii - np.dot(xii,wii))**2
        
    trainrmse = np.sqrt(np.mean(trainRMSE))
    
    ctrainRMSE = np.zeros((Nc,1))
    for kkk in range(Nc):
    #for kkk in range(len(W)):
        temp = trainRMSE[S[:-1]==(kkk+1)]
        ctrainRMSE[kkk,0] = np.sqrt(np.mean(temp))
        
    return ctrainRMSE, trainrmse

def rmsmallclusters(Nlon,Nlat,S0,W0,Nc0,AdjIndtrain):
        
    Nmin = 100 # minimum number of training data point required to be considered as a cluster
    S0 = S0[:-1,[0]]
    W0 = np.asarray(W0)
   
    nvtr, nktr = np.unique(S0[S0>0],return_counts=True) # [Nc,] nk is num of data in each cluster 
    
    indices = [i for i, v in enumerate(nktr) if v < Nmin]
    labels = nvtr[indices]
    
    S = np.copy(S0)
    #S = S.T # [N,1], training data cluster labels 

    Nc = Nc0-len(indices)
    W = np.delete(W0,indices,axis=0)
    nnind = AdjIndtrain.max() # not a valid neighbor index
    for label in labels:
        #ind1,ind2 = np.where(S0==label) # get indices of small clusters to be removed
        ind2,ind1 = np.where(S0==label) # get indices of small clusters to be removed
        for num in range(len(ind2)):
            if(ind2[num]<Nlon*Nlat):
                # need to deal with points with no training neighbours
                continue
            nind = AdjIndtrain[ind2[num],:]
            nind = nind[nind<nnind] # 2d indices of the neighbors
            nlabels = S[nind,0] # labels for all training neighbours
            nlabels = nlabels[nlabels!=label] # only count the labels for large clusters
            s = np.bincount(nlabels).argmax() # most frequent cluster label in the neighbour
            if(s>0):
                S[ind2,0] = s # select a new cluster label for this point
            else:
                print("ERROR! at " + str(ind2) + '\n')
    
    S = np.concatenate((S,[[-10]]),axis=0) # [N+1,1]
    W = W.tolist()

    return S,W,Nc





#### calculate test RMSE using past to predict future
def calTestRMSE3(S,W,AdjHtest,AdjIndtest,Ntest,Ntotal,Nc,Nn,N,Xtest,Ytest):
#    #### estimate Stest base on minimum residual error of \\y-WX\\  
    Stest = -10*np.ones((Ntest,1)) # [Ntest,] array, cluster index for test data points    
    Stest = Stest.astype(int) 
    Stest_est = -10*np.ones((Ntest,1))
   
    Stotal = np.concatenate((S[:-1,[0]],Stest,[[-10]]),axis=0) # [Ntotal+1,1], cluster index for train+test data    
    testRMSE = np.nan*np.ones((Ntest,1))
    WT = np.asarray(W).T  #[D,Nc] matrix ?
    for ii in range(Ntest):
        ## for existing cluster (k==1,2,...)  
        vech = AdjHtest[ii,:] # [Nn,] 1d array, f(dij) = GAMMA*exp(-dij)
        adjind = AdjIndtest[ii,:] # [Nn,], global 2d index for train+test neighbor
        
        tradjind = adjind[adjind<Ntotal] # 2d index of actual TRAIN+TEST neighbor
        if(len(tradjind)==0): # skip test data that has no (training) neighbors
            continue
        SSj = np.dot(Stotal[adjind],np.ones((1,Nc)))# extend to [Nn,Nc] matrix by repeating column
        tempk = np.arange(1,Nc+1).reshape((1,Nc))# [1,Nc]
        K = np.dot(np.ones((Nn,1)),tempk)# extend to [Nn,Nc] matrix by repeating row
        M = 1.0*(SSj==K) # [Nn,Nc]
        energy = np.exp(np.dot(vech,M)) # [Nc,] 1d array
        ## calculate the num of data points belong to cluster k
        nv, nk = np.unique(Stotal[Stotal>0],return_counts=True) # [Nc,] nk is num of data in each cluster  
        pQ = np.multiply(nk,energy).reshape((1,Nc)) # [Nc,1], probability of drawing from cluster k
        pQ = 1.0*pQ / np.sum(pQ) # normalize
       
        Stest_est[ii,0] = np.argmax(pQ,axis=1)[0]+1
        Stotal[N+ii,0] = Stest_est[ii,0]
       
        xii = Xtest[ii,:].T
        yii = Ytest[ii,0]        
        yQ = np.dot(xii.T,WT) # [1,Nc] possible ys for each cluster
        
        #### soft label
        yiihat = np.dot(yQ,pQ.T) # predicted yii
        testRMSE[ii,0] = (yii - yiihat)**2
        
        #### hard label
    #        pQ2 = pQ / np.max(pQ)
    #        pQ2[pQ2<1] = 0
    #        #PQ2[[ii],:] = pQ2
    #        yiihat2 = np.dot(yQ,pQ2.T)
    #        testRMSE[ii,0] = (yii - yiihat2)**2        
          
    testrmse = np.sqrt(np.nanmean(testRMSE))
    return testrmse, Stotal



def saveResults(resultpath,fold,nIter,Nmon,dataname,S,W,BETA,GAMMA,RHO,sigma1,
                sigma2,ALPHA,N,Nc,NcList,D,testrmse,trainrmse,repeat,Nlat,Nlon,
                Xinfotrain,Xinfotest,Ntest,Stotal,startTime=0):
    import os
    import time
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
        
    savename = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())+'-'+str(fold)+'-'+str(repeat)    
    savetime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
    ftxt = open(resultpath+savename+'.txt','w')
    ftxt.write('saved time: '+savetime+'\n')
    ftxt.write('saved after '+str(nIter)+'-th iteration\n')
    ftxt.write("total running time is " + str((time.time()-startTime)/60.0) + " minutes\n")   
    ftxt.write('total number of months: ' + str(Nmon) + '\n')
    ftxt.write('dataset: ' + dataname + '\n')
    ftxt.write('BETA='+str(BETA)+', GAMMA='+str(GAMMA)+', RHO='+str(RHO)+'\n')
    ftxt.write('sigma1='+str(sigma1)+', sigma2='+str(sigma2)+', ALPHA='+str(ALPHA)+'\n')
    ftxt.write('D='+str(D)+'; remaining number of clusters Nc='+str(Nc)+'\n\n')
    ftxt.write('total test RMSE = '+str(testrmse)+'\n')
    ftxt.write('total train RMSE = '+str(trainrmse)+'\n')
    #ftxt.write('train RMSE for each cluster is: \n')
    #ftxt.write(str(ctrainRMSE)+'\n\n')
    #ftxt.write('\n')
    ftxt.write('W =\n')
    #for k in range(Nc):
    for k in range(1,len(W)+1):
        for d in range(D):
            ftxt.write(str(W[k-1][d])+'  ')
        ftxt.write(str(np.count_nonzero(S==k)))
        ftxt.write('\n\n')
    ftxt.write('\n')
    ftxt.write('S=\n')    
    for i in range(N):
        ftxt.write(str(S[i,0])+'\n')    
    ftxt.close()

    #### label map not including estimated test data labels (0 for all test data)
    labelmap = -1*np.ones((Nlat,Nlon,Nmon)) # label -1 for missing data location
    for n in range(0,N):
        #[i,j,t] = Xinfotrain[n,2:5].astype(int) # dataprocess4 data type
        [i,j,t] = Xinfotrain[n,0:3].astype(int) # dataprocess5 data type
        labelmap[j,i,t] = S[n,0]
    for n in range(0,Ntest):
        #[i,j,t] = Xinfotest[n,2:5].astype(int)
        [i,j,t] = Xinfotest[n,0:3].astype(int)
        if(labelmap[j,i,t]!=-1):
            print("Error! train and test overlap!")
            exit()
        labelmap[j,i,t] = 0 # label 0 for test data
        
    #### S label map including estimated test data labels
    Xinfototal = np.concatenate((Xinfotrain,Xinfotest),axis=0)
    Smap = -1*np.ones((Nlon,Nlat,Nmon)) # -1 means missing data
    for ii in range(len(Stotal[:-1,0])):
        #[i,j,t] = Xinfototal[ii,2:5].astype(int)
        [i,j,t] = Xinfototal[ii,0:3].astype(int)
        Smap[i,j,t] = Stotal[ii,0]

    import scipy.io as sio
    result = {}
    result['W'] = W
    result['S'] = S[:-1,0]
    result['labelmap'] = labelmap
    result['sigma1'] = sigma1
    result['sigma2'] = sigma2
    result['ALPHA'] = ALPHA
    result['Nc'] = Nc
    result['BETA'] = BETA
    result['GAMMA'] = GAMMA
    result['RHO'] = RHO
    result['testrmse'] = testrmse
    result['trainrmse'] = trainrmse
    result['dataname'] = dataname
    result['savedtime'] = savetime
    result['iterations'] = nIter+1
    result['runtime'] = (time.time()-startTime)/60.0
    result['Stotal'] = Stotal[:-1,0]
    result['Smap'] = Smap # -1 means missing data
    result['NcList'] = NcList # number of clusters in each iteration
    sio.savemat(resultpath+savename + '.mat', result) # save to python project folder
    #sio.savemat('/home/yumin/myProgramFiles/myMATLABFiles/MRFmodel/results/'+savename+'.mat', result) # save to matlab project folder 


def plot_3d_fig(Nlon,Nlat,Nmon,Smap):
    #### plot 3d figure
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    ax = Axes3D(fig)
    for i in range(Nlon):
        for j in range(Nlat):
            for t in range(Nmon):
                if(Smap[i,j,t]==1):
                    cluster1 = ax.scatter(i,j,t,c='r',marker='o',alpha=0.5,label='cluster 1')
                elif(Smap[i,j,t]==2):
                    cluster2 = ax.scatter(i,j,t,c='g',marker='.',alpha=0.5,label='cluster 2')
                elif(Smap[i,j,t]==3):
                    cluster3 = ax.scatter(i,j,t,c='b',marker='+',alpha=0.5,label='cluster 3')
                      
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('month')
    ax.set_title('clustering results of synthetic data')
    ax.legend(handles=[cluster1,cluster2,cluster3])


def cal_nmi(Stotal,Strue,Ntotal):
    #### for synthetic data
    Spred = Stotal[:-1,0].reshape((Ntotal,))
    Strue = Strue.reshape((Ntotal,))
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(Spred,Strue)
    return nmi

def cal_Smap(Xinfotrain,Xinfotest,Nlon,Nlat,Nmon,Stotal):
    #### S label map including estimated test data labels
    Xinfototal = np.concatenate((Xinfotrain,Xinfotest),axis=0)
    Smap = -1*np.ones((Nlon,Nlat,Nmon))
    for ii in range(len(Stotal[:-1,0])):
        #[i,j,t] = Xinfototal[ii,2:5].astype(int)
        [i,j,t] = Xinfototal[ii,0:3].astype(int)
        Smap[i,j,t] = Stotal[ii,0]
    return Smap