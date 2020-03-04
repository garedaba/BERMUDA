#!/usr/bin/env python
# coding: utf-8

# create code to generate and run on synthetic data
import torch
import os

import numpy as np
import pandas as pd
import random
import time
import logging
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from modules.data_generation import generate_data, plot_synthetic_data
from modules.helpers import *

from BERMUDA import training, testing

# CUDA setup
device_id = 0 # ID of GPU to use
cuda = torch.cuda.is_available()
print('GPU is available: ', cuda)
if cuda:
    torch.cuda.set_device(device_id)

# model parameters
code_dim = 2 # latent space
batch_size = 100 # batch size for each cluster
num_epochs = 1500
base_lr = 1e-3
lr_step = 200  # step decay of learning rates
l2_decay = 5e-5
gamma = 1  # regularization between reconstruction and transfer learning - changes with epoch
log_interval = 1

# parameter dictionary
nn_paras = {'code_dim': code_dim, 'batch_size': batch_size, 'num_epochs': num_epochs,
            'base_lr': base_lr, 'lr_step': lr_step,
            'l2_decay': l2_decay, 'gamma': gamma,
            'cuda': cuda, 'log_interval': log_interval}


# data generation parameters
number_of_subjects = 5
number_of_tissues = 10
number_of_voxels = 150
number_of_features = 10
set_noise = 0.35

pre_process_paras = {'scale': False,          # Z-score
                     'standardise': True}   # [0,1]

outDir = 'synthetic_data'
os.makedirs(outDir, exist_ok=True)


plt.ioff()

if __name__ == '__main__':

    # SYNTHETIC DATA ########################################################################
    # generate synthetic data
    print('')
    print('generating data')
    metadata, data = generate_data(n_subjects = number_of_subjects,
                                   n_tissue_types = number_of_tissues,
                                   n_voxels = number_of_voxels,
                                   n_features = number_of_features,
                                   noise = set_noise)

    plot_synthetic_data(metadata, data, outdir=outDir)
    #########################################################################################

    # DATA PREP #############################################################################
    # split into train and test
    train_idx, test_idx = train_test_split(np.unique(metadata['subjects']), test_size=0.2, shuffle=True, random_state=42)
    x_train, x_test = data[metadata['subjects'].isin(train_idx)], data[metadata['subjects'].isin(test_idx)]
    y_train, y_test = metadata[metadata['subjects'].isin(train_idx)], metadata[metadata['subjects'].isin(test_idx)]

    # remove some tissue types from some subjects in training data
    x_train, y_train = decimate_data(x_train, y_train, number_of_tissues)

    # scale/standardise
    x_train, x_test = pre_process_datasets(x_train, x_test, pre_process_paras)

    # get cluster pairs, set up X data for BERMUDA
    dataset_list, _, cluster_pairs = prepare_data(x_train, y_train)
    #########################################################################################

    # MODEL TRAINING ########################################################################
    # set seed for reproducibility
    #seed = 0
    #torch.manual_seed(seed) # set seed on CPU and GPU for reproducibility
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    nn_paras['num_inputs'] = np.shape(dataset_list[0]['data'])[1]

    # training
    model, loss_total_list, loss_reconstruct_list, loss_transfer_list = training(dataset_list, cluster_pairs, nn_paras)

    # plot training loss
    plot_loss(loss_total_list, loss_reconstruct_list, loss_transfer_list, outDir + '/model-loss.png')

    # training code
    code_list, recon_list = testing(model, dataset_list, nn_paras)
    train_code = (np.concatenate(code_list, axis=1).transpose())
    train_recon = (np.concatenate(recon_list, axis=1).transpose())
    ##########################################################################################

    # PLOT LATENT SPACE ######################################################################
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.scatter(train_code[:,0], train_code[:,1], c=y_train.subjects,  cmap='gist_rainbow')
    ax2.scatter(train_code[:,0], train_code[:,1], c=y_train.tissues,  cmap='rainbow')
    ax1.set_title('subject')
    ax2.set_title('tissue')
    for ax in [ax1, ax2]:
        ax.set_xlabel('ae 1')
        ax.set_ylabel('ae 2')

    plt.savefig(outDir + '/training-latent-space.png')
    ##########################################################################################

    # EVALUATE TEST DATA #####################################################################
    test_data_set, _, _ = prepare_data(x_test, y_test)

    code_list_test, recon_list_test = testing(model, test_data_set, nn_paras)
    test_code = (np.concatenate(code_list_test, axis=1).transpose())
    test_recon = (np.concatenate(recon_list_test, axis=1).transpose())


    train_mse = mean_squared_error(x_train, train_recon)
    test_mse = mean_squared_error(x_test, test_recon)
    print('train_error {:.3f}'.format(train_mse))
    print('test_error {:.3f}'.format(test_mse))

    # align to latent space
    all_aligned =[]

    for n, test_sub in enumerate(pd.unique(y_test.subjects)):
        align_sub_code, _, _ = align_latent_space(y_test[y_test.subjects==test_sub], y_train, test_code[y_test.subjects==test_sub], train_code)
        all_aligned.append(align_sub_code)

    all_aligned = np.vstack(all_aligned)

    # plot
    fig, (ax1, ax2) = plt.subplots(2,3, figsize=(12,6), sharey=True, sharex=True)
    ax1[0].scatter(train_code[:,0], train_code[:,1], c=y_train.subjects, cmap='gist_rainbow')
    ax2[0].scatter(train_code[:,0], train_code[:,1], c=y_train.tissues, cmap='rainbow')

    ax1[1].scatter(test_code[:,0], test_code[:,1], c=y_test.subjects, cmap='gist_rainbow')
    ax2[1].scatter(test_code[:,0], test_code[:,1], c=y_test.tissues, cmap='rainbow')

    ax1[2].scatter(all_aligned[:,0], all_aligned[:,1], c=y_test.subjects, cmap='gist_rainbow')
    ax2[2].scatter(all_aligned[:,0], all_aligned[:,1], c=y_test.tissues, cmap='rainbow')

    ax1[0].set_title('training data')
    ax1[1].set_title('test data')
    ax1[2].set_title('aligned data')

    ax1[0].set_ylabel('by subject')
    ax2[0].set_title('by tissue')

    plt.savefig(outDir + '/latent-spaces-all.png')

    ##########################################################################################

