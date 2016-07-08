"""the second step of knn backbone clustering"""
#oAuthor:bruc14@163.com

import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from numpy import genfromtxt, savetxt
import numpy as np
import argparse
import pdb
from time import time
import os
import knnclusterstep1
sys.path.append("..")
import cm
from sklearn.metrics.cluster import normalized_mutual_info_score
import settings
#cm is a python module writed for this algorithm

######################################################################################
# The process of the second step


def _FSDPclustering(NNpoints,densities):
    """The function of the second step of this algorithm.

    Parameters
    ----------
    NNpoints : array-like or matrix, shape (the number of clusters after the first step process, the top k of nearest neighbors)
        The nearest neighbors matrix.

    densities : list, length of n_samples.
        The sum distance of the top k nearest neighbors

    Returns
    -------
    cluster : the result of clustering, which is a 2D list.

    """

    NNpointsDensities=np.hstack((NNpoints,densities.reshape(len(densities),1)))
    indexSorted=np.argsort(densities)
    highdensityPoints=NNpointsDensities[indexSorted][:int(len(NNpoints)*settings.getTop_High_Density()),:len(NNpointsDensities)-1]
    lowdensityPoints=NNpointsDensities[indexSorted]

    # clustering the first half density points using mutual k nearest neighbors
    cluster=[]
    for nnpoint in highdensityPoints:
        cluster.append([])
        for nn in nnpoint:
            if nn in highdensityPoints[:,0]:
                cluster[len(cluster)-1].append(nn)
    cluster=cm.mergeCluster1(cluster)

    # after the cluster backbone has been found, each remaining point is assigned to the same cluster as its nearest neighbor of higher density.
    for i in range(1,len(lowdensityPoints)+1):
        ndim=lowdensityPoints.shape[1]
        cluster.append([])
        cluster[len(cluster)-1].append(lowdensityPoints[-i][0])
        for PointDensity in lowdensityPoints[-i][1:ndim-1]:
            # pdb.set_trace()
            index=np.argwhere(lowdensityPoints[:,0]==PointDensity)
            if index.size!=0:
                index=index[0][0]
                if lowdensityPoints[-i][-1]>lowdensityPoints[index][-1]:
                    cluster[len(cluster)-1].append(PointDensity)
                    break
    cluster=cm.mergeCluster1(cluster)

    cluster=cm.simplifyCluster(cluster)
    return cluster

def CLUB (X,y,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    # pdb.set_trace()
    neigh.fit(X, y)
    neigh_dist, neigh_ind = neigh.kneighbors(X)
    neigh_ind=neigh_ind[:,1:]
    neigh_dist=neigh_dist[:,1:]

    nncluster = knnclusterstep1.NNcluster()
    # clusters,s=nncluster.clustering(neigh_ind[:,:9])
    clusters,s=nncluster.clustering(neigh_ind[:,:k/2])
    theclu=[]
    for cluster in clusters:
        densities = np.mean(neigh_dist[cluster, :k/2],axis=1)
        clu=_FSDPclustering(np.hstack((np.array(cluster).reshape(len(cluster),1),neigh_ind[cluster])),densities)
        for cl in clu:
            theclu.append(cl)

    # theclu,yichangdian = selectOutliers(neigh_dist,theclu)
    return theclu
    # return theclu,yichangdian

def selectOutliers (neigh_dist,clusters):
    global TOP_HIGH_DENSITY
    outliers=[]
    clustersWithoutOutlier=[]
    for cluster in clusters:
        PdensityDevideOne=np.hstack((np.array(cluster).reshape(len(cluster),1),np.mean(neigh_dist[cluster],axis=1).reshape(len(cluster),1)))
        mean=PdensityDevideOne[:,1].mean()
        std=PdensityDevideOne[:,1].std()
        boundary = mean + 3*std
        clu=[]
        for item in PdensityDevideOne:
            if item[1]>boundary:
                outliers.append(item[0])
            else:
                clu.append(item[0])
        if clu:
            clustersWithoutOutlier.append(clu)
    return clustersWithoutOutlier,outliers


def CLUB_file (dataset,k):
    L = genfromtxt(open(dataset,'r'),dtype=float, delimiter='\t')
    if dataset.find('/')!=-1:
        filename=dataset[dataset.find('/'):dataset.find('.')]+'K'+str(k)+'TopDensity'+str(settings.getTop_High_Density())+os.path.splitext(dataset)[1]
    else:
        filename=os.path.splitext(dataset)[0]+'K'+str(k)+'TopDensity'+str(settings.getTop_High_Density())+os.path.splitext(dataset)[1]

    L=np.array(L)
    y=L[:,0]
    y=y.astype(int)
    X = L[:,1:]
    theclu=CLUB(X,y,k)
    # theclu,outliers=CLUB(X,y,k)

    # if outliers:
        # theclu.append(outliers)
    cm.plotFigure(X,theclu,'result/'+filename+'Cluster')
    # print theclu
    plabel={}
    for i,cl in enumerate(theclu):
        for e in cl:
            plabel[e]=i

    # for e in outliers:
        # plabel[e]=-1

    plabel=plabel.values()
    if 'temp' in os.listdir('.'):
      pass
    else:
      os.mkdir('temp')
    f=open('temp/'+filename+'.plabel','w')
    f.write(str(np.ndarray.tolist(y))+'\n')
    f.write(str(str(plabel)))
    f.close()

    # Get ARI score and running time, and write them into a file
    if 'result' in os.listdir('.'):
      pass
    else:
      os.mkdir('result')
    f=open('result/'+filename+'.score','w')
    ARI,PRFS=cm.getScore(plabel,y)
    nmi=normalized_mutual_info_score(plabel,y)
    # pdb.set_trace()
    ARIstr,PRFSstr=str(ARI),PRFS
    f.write('ARI:'+ARIstr)
    f.write('\nNMI:'+str(nmi))
    f.write('\n\n'+PRFSstr)

def normalize (dataset,normalizationMethod='None'):
    L = genfromtxt(open(dataset,'r'),dtype=float, delimiter='\t')
    L=np.array(L)
    y=L[:,0]
    y=y.astype(int)
    X = L[:,1:]
    if normalizationMethod == 'Norm01':
        # pdb.set_trace()
        if (X.max(axis=0) != X.min(axis=0)).all():
            X=(X-X.min())/(X.max()-X.min())
        else:
            iseq=(X.max(axis=0) != X.min(axis=0))
            iseq=list(iseq)
            index=iseq.index(False)
            X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
            X[:,index]=0
        filename = os.path.splitext(dataset)[0] + '.norm01'
    elif normalizationMethod == 'Zscore':
        if (X.max(axis=0) != X.min(axis=0)).all():
            X=(X-X.mean(axis=0))/X.std(axis=0)
        else:
            iseq=(X.max(axis=0) != X.min(axis=0))
            iseq=list(iseq)
            index=iseq.index(False)
        # pdb.set_trace()
            X=(X-X.mean(axis=0))/X.std(axis=0)
            X[:,index]=0
        filename = os.path.splitext(dataset)[0] + '.zscore'
    elif normalizationMethod == 'None':
        filename = dataset

    L=np.hstack((y.reshape(len(y),1),X))
    np.savetxt(filename, L, fmt='%.8f', delimiter='\t')

def labelNorm (dataset):
    L = genfromtxt(open(dataset,'r'),dtype=float, delimiter='\t')
    L=np.array(L)
    y=L[:,0]
    X = L[:,1:]
    label=set(y)
    labelTable={}
    i=0
    for e in label:
        labelTable[e]=i
        i+=1

    yNormed=[]
    for e in y:
        yNormed.append(labelTable[e])
    yNormed=np.array(yNormed)
    
    L=np.hstack((yNormed.reshape(len(yNormed),1),X))
    np.savetxt(dataset, L, fmt='%.8f', delimiter='\t')

def zscoreNormalize (dataset):
    L = genfromtxt(open(dataset,'r'),dtype=float, delimiter='\t')
    L=np.array(L)
    y=L[:,0]
    y=y.astype(int)
    X = L[:,1:]
    X=(X-X.mean(axis=0))/X.var()**0.5
    L=np.hstack((y.reshape(len(y),1),X))
    np.savetxt(dataset+'.zscore',L,fmt='%.8f',delimiter='\t')
    
##################################################################################
#A script using this module
if __name__=='__main__':
    for k in range(65, 90):
        CLUB_file('datasets8/t4.in',k=k)
        CLUB_file('datasets8/t5.in',k=k)
    # normalize('data/ecoli.in', 'Norm01')
