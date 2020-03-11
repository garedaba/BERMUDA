#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def pre_process_datasets(train, test, paras):
    """perform Z-score standardisation or min-max scaling (to [0,1])
    scaling/standardisation based on parameters of the training data

    train: train data
    test: test data
    paras: processing parameters

    return:
    train, test: scaled/standardised data
    """

    if paras['scale']:
        ss = StandardScaler()
        train, test = ss.fit_transform(train), ss.transform(test)
    if paras['standardise']:
        mm = MinMaxScaler()
        train, test = mm.fit_transform(train), mm.transform(test)

    return train, test


def decimate_data(data, metadata, n_tissues):
    """remove a random selection of tissues from each subject in the training
    data when using synthetically generated data, to mimic characteristics
    of real data where not all subjects will have all tissue types

    data: array/DataFrame, synthetic data to be decimated
    metadata: pandas DataFrame with synthetic metadata: subject ID, tissue ID

    returns:
    data, metadata: with randomly selected tissue types removed
    """

    indices = pd.DataFrame()
    for s in pd.unique(metadata['subjects']):
        tissues_to_keep = np.random.choice(np.unique(metadata['tissues']), np.round(n_tissues*.75).astype(int))
        s_indices = (metadata['subjects']==s) & (metadata['tissues'].isin(tissues_to_keep))

        indices = pd.concat((indices, s_indices), axis=1)
    indices = indices.sum(axis=1)

    return data[indices==1], metadata[indices==1]


def prepare_data(x, y):
    """set up data ready for BERMUDA.
    identify matched tisse types across subjects and output cluster pairs and subject-specific numbers
    create list of datasets, one per 'cluster'

    x: data array for training
    y: metadata, pandas DataFrame

    returns:
    dataset_list: list of datasets, one per tissue type per subject
    y: updated metadata with tissue x subject labels
    cluster_pairs: matched tissue types across subjects

    """
    # get unique ID for each cell type across subjects
    tissue_subject_data=np.zeros((len(y)))
    for ind, (sub, tis) in enumerate(pd.unique(list(zip(y.subjects, y.tissues)))):
        tissue_subject_data[(y['subjects']==sub) & (y['tissues']==tis)] = ind
    y.loc[:,'tissue_by_subject'] = tissue_subject_data.copy()

    # create list of cluster pairs - identifying cells that are present in more than one subject
    new = []
    for ind1, t in enumerate(y.tissues):
        for ind2, u in enumerate(y.tissues):
            if t==u: #i.e.: if the tissues are the same
                # collate subject specific tissue IDs instead
                new_t = y.tissue_by_subject.iloc[ind1]
                new_u = y.tissue_by_subject.iloc[ind2]
                # as long as they are not from the same subject
                if new_t != new_u:
                    # append sorted tuple (to make sure we don't double up: (1,2)==(2,1))
                    new.append(tuple(sorted((new_t, new_u))))
    # only unique pairs
    pairs = pd.unique(new)
    pairs = np.vstack(pairs)

    # cluster pairs for BERMUDA
    cluster_pairs = np.hstack((pairs, np.ones((len(pairs))).reshape(-1,1)))

    # break down data into lists, one per cell
    dataset_list = []
    for s in pd.unique(y['subjects']):
        dataset = {}
        dataset['data'] = x[y.subjects==s]
        dataset['tissue_labels'] = y.loc[y.subjects==s, 'tissue_by_subject'].values.astype(int)
        dataset['sample_labels'] = y.loc[y.subjects==s, 'subjects'].values.astype(int)

        dataset_list.append(dataset)

    return dataset_list, y, cluster_pairs


def get_centroids(coordinates):
    """ calculate centroid of set of n coordinates
    coordinates: nD array (x,y,z,...,n) of coordinates

    returns:
    centroids: list of centroids
    """
    n_dims = coordinates.shape[1]
    length = coordinates.shape[0]

    centroids = []
    for d in np.arange(n_dims):
        centroids.append(np.sum(coordinates[:,d]/length))

    return centroids


def rotation_and_translation(P, Q):
    """Umeyama algorithm to calculate rot and trans between point sets
    https://gist.github.com/nh2/bc4e2981b0e213fefd4aaa33edfb3893

    P: k x d array, data to align
    Q: k x d array, target data

    return:
    c, R, T: such that aligned_P = P.dot(c*R)+T
    """
    assert P.shape == Q.shape
    n, dim = P.shape

    centredP = P - P.mean(axis=0)
    centredQ = Q - Q.mean(axis=0)

    C = np.dot(centredP.T, centredQ) / n
    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)
    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor
    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t

def align_latent_space(y_test, y_train, test_code, train_code, remove_labels=True):
    """ align codes to latent space learned during training
    y_test: array, test metadata - unseen by algorithm
    y_train: array, metadata used in model training
    test_code: coordinates of test data in latent space
    train_code: coordinate of train data in latent space

    returns:
    rotated codes, rotation matrix, translation matrix
    such that rotated_codes = codes.dot(rotation)+translation
    """
    test_coord = []
    train_coord = []
    # for each tissue in test data
    if remove_labels:
        test_labels = pd.unique(np.random.choice(pd.unique(y_test.tissues), size=len(pd.unique(y_test.tissues))))
        print('{:} labels removed'.format(len(pd.unique(y_test.tissues))-len(test_labels)))
    else:
        test_labels = y_test.tissues.copy()

    for tiss in pd.unique(y_test.tissues):
        # get centroids for test data
        test_coord.append(get_centroids(test_code[y_test.tissues==tiss]))
        # corresponding centroids in training data
        train_coord.append(get_centroids(train_code[y_train.tissues==tiss]))

    test_coord = np.vstack(test_coord)
    train_coord = np.vstack(train_coord)

    # calculate rotation and translation in native space
    sc, rot, trans = rotation_and_translation(test_coord, train_coord)

    # rotate test data to training latent space
    rot_test_code = test_code.dot(sc*rot) + trans

    return rot_test_code, (sc*rot), trans


def plot_loss(loss_total_list, loss_reconstruct_list, loss_transfer_list, save_path):
    """ Plot loss versus epochs
    loss_total_list: list of total loss
    loss_reconstruct_list: list of reconstruction loss
    loss_transfer_list: list of transfer loss
    save_path: path to save the plot

    returns:
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    ax1.plot(range(len(loss_total_list)), loss_total_list, "r:",linewidth=1)
    ax1.legend(['total loss'])
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax2.plot(range(len(loss_reconstruct_list)), loss_reconstruct_list, "b--",linewidth=1)
    ax2.legend(['reconstruction loss'])
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax3.plot(range(len(loss_transfer_list)), loss_transfer_list, "g-",linewidth=1)
    ax3.legend(['transfer loss'])
    ax3.set_xlabel("epochs")
    ax3.set_ylabel("loss")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def project_to_orthogonal_axes(codes, ndim=None):
    """ project embedded data to orthogonal axies via svd
    codes: n x d embedded data
    ndim: if None, return all dimensions, otherwise if ndim < d return reduced data

    returns:
    projected_code: data projected onto orth. axes
    transformation: transformation matrix
    """

    dim = np.shape(codes)[1]
    if ndim is None:
        d = dim
    elif ndim<dim:
        d = ndim
    else:
        print("incorrect number of dimensions specified!")
        exit(1)

    # perform svd
    u, s, vt = np.linalg.svd(codes)

    # transformed x
    transformed_code = u[:,:d].dot(np.diag(s[:d]))

    # transformer
    transformer = vt[:3,:].T

    return transformed_code, transformer
