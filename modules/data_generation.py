#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
import rpy2.robjects as ro
from rpy2.robjects.pandas2ri import ri2py
from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import matplotlib.pyplot as plt

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky
    https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def generate_data(n_subjects=10, n_tissue_types=15, n_voxels=50, n_features=10, noise=0.5):
    """ Generates correlated feature data with covariance structure dictated by subject and tissue type """

    # total number of observations
    total_n = n_subjects * n_tissue_types * n_voxels
    print("total number of observations: ", total_n)

    # subject IDS
    subjects = np.repeat(np.arange(n_subjects), n_tissue_types*n_voxels)
    # tissue IDS
    tissues = np.repeat(np.tile(np.arange(n_tissue_types), n_subjects), n_voxels)

    # One hot encode
    meta = np.vstack((subjects, tissues)).T
    meta_ohe = OneHotEncoder().fit_transform(meta).toarray()

    # calculate covariance matrix and project to SPD
    label_data = zscore(meta_ohe, axis=0)
    label_data_noisy = label_data + np.random.normal(loc=0, scale=noise, size=np.shape(label_data))
    covMat = np.dot(label_data_noisy, label_data_noisy.T)
    covMat = nearestPD(covMat)

    # use R function to create correlated feature data
    rstring = '''suppressMessages(library(simstudy))
                 suppressMessages(library(clusterGeneration))
                 suppressMessages(library(Matrix))

                 corrmat <- rcorrmatrix('''+str(n_features)+''', alphad = 0.1)
                 dims<-dim(corrmat)
                 corrmat <- array((nearPD(corrmat, corr=TRUE)$mat), dim=dims)
                 corr_Data <- genCorData('''+str(total_n)+''', mu = rep(0,'''+str(n_features)+'''), sigma = 1, corMatrix=corrmat)
                 colnames(corr_Data)[1] <- c('vox')
                 corr_Data
              '''
    corr_Data = ro.r(rstring)

    # remove voxel ID, translate to python
    data = ri2py(corr_Data)
    data = data.iloc[:,1:]
    
    # impart covariance structure dictated by meta data
    data = (data.T.dot(np.linalg.cholesky(covMat).T)).T
    
    meta = pd.DataFrame(meta)
    meta.columns=['subjects', 'tissues']
    
    return meta, data

def plot_synthetic_data(meta, data, transform='pca', outdir='.'):
    """Plot a low-d representation of synthetic data
    transform: 'pca', 'umap', or 'tsne'
    """
    
    if transform=='pca':
        transformer = PCA(n_components=2)
    elif transform=='umap':
        transformer = UMAP(n_components=2)
    elif transform=='tsne':
        transformer = TSNE(n_components=2)
    else:
        print("transformer not recognised - specify 'pca', 'umap' or 'tsne'")
        exit(1)
    
    ss = StandardScaler()
    
    # project to 2D
    lowd = transformer.fit_transform(ss.fit_transform(data))
    
    # plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), sharey=True)
    ax1.scatter(lowd[:,0], lowd[:,1], c=meta['subjects'], cmap='gist_rainbow')
    ax2.scatter(lowd[:,0], lowd[:,1], c=meta['tissues'], cmap='rainbow')

    ax1.set_title('subjects', fontsize=20)
    ax2.set_title('tissues', fontsize=20)

    for ax in [ax1, ax2]:
        ax.set_xlabel('dim1', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
    ax1.set_ylabel('dim1', fontsize=15)

    
    plt.tight_layout()
    plt.savefig(outdir + '/' + transform + '-synthetic.png')
    plt.show()
