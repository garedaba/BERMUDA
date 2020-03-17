#!/usr/bin/env python
# coding: utf-8

# create code to generate and run on synthetic data
import torch
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, plot_confusion_matrix
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import LabelEncoder

from modules.data_generation import generate_data, plot_synthetic_data
from modules.helpers import *
from models.classification import *

from BERMUDA import training, testing

# SETUP ########################################################################
# load parameters
with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

print(yaml.dump(cfg, default_flow_style=False, default_style=''))
print('')

# CUDA setup
cuda = torch.cuda.is_available()
print('GPU is available: ', cuda)
if cuda:
    torch.cuda.set_device(cfg['cuda_setup']['device_id'])

# model parameters
nn_params = cfg['model_params']
nn_params['cuda'] = cuda

# data generation parameters
data_gen_params = cfg['data_gen_params']

# preprocessing
pre_process_paras = cfg['pre_process_paras']

# classification
le = LabelEncoder()
kern = 1. * RBF()
gpc = GaussianProcessClassifier(kernel=kern, n_jobs=-2)

# output
outDir = 'synthetic_data'
os.makedirs(outDir, exist_ok=True)

plt.ioff()

if __name__ == '__main__':

    # SYNTHETIC DATA ########################################################################
    # generate synthetic data
    print('')
    print('generating data')
    metadata, data = generate_data(n_subjects = data_gen_params['number_of_subjects'],
                                   n_tissue_types = data_gen_params['number_of_tissues'],
                                   n_voxels = data_gen_params['number_of_voxels'],
                                   n_features = data_gen_params['number_of_features'],
                                   noise = data_gen_params['set_noise'])

    plot_synthetic_data(metadata, data, transform='pca', outfile=outDir + '/pca-synthetic.png')
    #########################################################################################

    # DATA PREP #############################################################################
    # split into train and test
    train_idx, test_idx = train_test_split(np.unique(metadata['subjects']), test_size=0.2, shuffle=True, random_state=42)
    x_train, x_test = data[metadata['subjects'].isin(train_idx)], data[metadata['subjects'].isin(test_idx)]
    y_train, y_test = metadata[metadata['subjects'].isin(train_idx)], metadata[metadata['subjects'].isin(test_idx)]

    # remove some tissue types from some subjects in training data
    x_train, y_train = decimate_data(x_train, y_train, data_gen_params['number_of_tissues'])

    # scale/standardise
    x_train, x_test = pre_process_datasets(x_train, x_test, pre_process_paras)

    # get cluster pairs, set up X data for BERMUDA
    dataset_list, _, cluster_pairs = prepare_data(x_train, y_train)
    #########################################################################################

    # MODEL TRAINING ########################################################################
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nn_params['num_inputs'] = np.shape(dataset_list[0]['data'])[1]

    # training
    model, loss_total_list, loss_reconstruct_list, loss_transfer_list = training(dataset_list, cluster_pairs, nn_params)

    # plot training loss
    plot_loss(loss_total_list, loss_reconstruct_list, loss_transfer_list, outDir + '/model-loss.png')

    # training code
    code_list, recon_list = testing(model, dataset_list, nn_params)
    train_code = (np.concatenate(code_list, axis=1).transpose())
    train_recon = (np.concatenate(recon_list, axis=1).transpose())
    ##########################################################################################

    # EVALUATE TEST DATA #####################################################################
    test_data_set, _, _ = prepare_data(x_test, y_test)

    code_list_test, recon_list_test = testing(model, test_data_set, nn_params)
    test_code = (np.concatenate(code_list_test, axis=1).transpose())
    test_recon = (np.concatenate(recon_list_test, axis=1).transpose())

    train_mse = mean_squared_error(x_train, train_recon)
    test_mse = mean_squared_error(x_test, test_recon)
    print('')
    print('reconstruction error')
    print('train_error {:.3f}'.format(train_mse))
    print('test_error {:.3f}'.format(test_mse))
    print('')

    # align to latent space
    print('aligning latent spaces')
    all_aligned =[]

    for n, test_sub in enumerate(pd.unique(y_test.subjects)):
        align_sub_code, _, _ = align_latent_space(y_test[y_test.subjects==test_sub], y_train, test_code[y_test.subjects==test_sub], train_code)
        all_aligned.append(align_sub_code)

    all_aligned = np.vstack(all_aligned)

    # project data to the most discriminant axes
    rot_train_code, transformer = project_to_discriminant_axes(train_code, y_train.tissues, ndim=None)
    rot_test_code = transformer.transform(test_code)
    rot_all_aligned = transformer.transform(all_aligned)
    ##########################################################################################

    # SAVE OUTPUT ###############################################################################
    train_out = pd.concat((y_train, pd.DataFrame(rot_train_code, index=y_train.index)), axis=1)
    train_out.to_csv(outDir + '/rotated-embedded-train-data.csv')
    test_out = pd.concat((y_test, pd.DataFrame(rot_all_aligned, index=y_test.index)), axis=1)
    test_out.to_csv(outDir + '/rotated-embedded-test-data.csv')
    #########################################################################################

    # RUN CLASSIFICATION IN LATENT SPACE #########################################################
    # fit - use le to account for potential missing classes in test data
    trained_model = train_classifier(rot_train_code, le.fit_transform(y_train.tissues), gpc)
    # predict
    train_predicted_proba = trained_model.predict_proba(rot_train_code)
    test_predicted_proba = trained_model.predict_proba(rot_all_aligned)
    # get accuracies
    train_logloss, train_accuracy, train_confusion = calculate_model_accuracy(le.transform(y_train.tissues), train_predicted_proba)
    test_logloss, test_accuracy, test_confusion = calculate_model_accuracy(le.transform(y_test.tissues), test_predicted_proba)

    print('')
    print('training data:')
    print('accuracy: {:.3f} log loss: {:.3f}'.format(train_accuracy, train_logloss))
    print('')
    print('test data')
    print('accuracy: {:.3f} log loss: {:.3f}'.format(test_accuracy, test_logloss))

    # train using full data (not embedded)
    gpc2 = GaussianProcessClassifier(kernel=kern, n_jobs=-2)
    full_trained_model = train_classifier(x_train, le.fit_transform(y_train.tissues), gpc2)
    # get accuracies
    full_test_logloss, full_test_accuracy, _ = calculate_model_accuracy(le.transform(y_test.tissues), full_trained_model.predict_proba(x_test))

    print('')
    print('test data - no embedding')
    print('accuracy: {:.3f} log loss: {:.3f}'.format(full_test_accuracy, full_test_logloss))

    print('')
    ###############################################################################################

    # PLOTTING #####################################################################################
    # PLOT LATENT SPACE with training data #####################################################
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.scatter(train_code[:,0], train_code[:,1], c=y_train.subjects,  alpha=0.5, edgecolor='grey', s=30, cmap='jet')
    ax2.scatter(train_code[:,0], train_code[:,1], c=y_train.tissues,  alpha=0.5, edgecolor='grey', s=30, cmap='viridis')
    ax1.set_title('subject')
    ax2.set_title('tissue')
    for ax in [ax1, ax2]:
        ax.set_xlabel('ae 1')
        ax.set_ylabel('ae 2')

    plt.savefig(outDir + '/training-latent-space.png')
    ##########################################################################################

    # PLOT ALL LATENT SPACES #################################################################
    fig, (ax1, ax2) = plt.subplots(2,3, figsize=(12,6), sharey=True, sharex=True)
    ax1[0].scatter(train_code[:,0], train_code[:,1], c=y_train.subjects, alpha=0.5, edgecolor='grey', s=20, cmap='jet')
    ax2[0].scatter(train_code[:,0], train_code[:,1], c=y_train.tissues, alpha=0.5, edgecolor='grey', s=20, cmap='viridis')

    ax1[1].scatter(test_code[:,0], test_code[:,1], c=y_test.subjects,alpha=0.5, edgecolor='grey', s=20, cmap='jet')
    ax2[1].scatter(test_code[:,0], test_code[:,1], c=y_test.tissues, alpha=0.5, edgecolor='grey', s=20, cmap='viridis')

    ax1[2].scatter(all_aligned[:,0], all_aligned[:,1], c=y_test.subjects, alpha=0.5, edgecolor='grey', s=20, cmap='jet')
    ax2[2].scatter(all_aligned[:,0], all_aligned[:,1], c=y_test.tissues, alpha=0.5, edgecolor='grey', s=20, cmap='viridis')

    ax1[0].set_title('training data')
    ax1[1].set_title('test data')
    ax1[2].set_title('aligned data')

    ax1[0].set_ylabel('by subject')
    ax2[0].set_title('by tissue')

    plt.savefig(outDir + '/latent-spaces-all.png')
    ##########################################################################################

    # PLOT CONFUSION MATRICES ################################################################
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,4), )
    plot_confusion_matrix(trained_model, rot_train_code,  le.transform(y_train.tissues), values_format='1', cmap='Greens', ax=ax1)
    plot_confusion_matrix(trained_model, rot_all_aligned,  le.transform(y_test.tissues), values_format='1', cmap='Greens', ax=ax2)
    plot_confusion_matrix(full_trained_model, x_test,  le.transform(y_test.tissues), values_format='1', cmap='Greens', ax=ax3)
    ax1.set_title('embedded training data')
    ax2.set_title('embedded testing data')
    ax3.set_title('full testing data')

    plt.savefig(outDir + '/confusion-matrices.png')
    ##########################################################################################
