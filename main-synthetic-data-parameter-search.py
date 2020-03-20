#!/usr/bin/env python
# coding: utf-8

# create code to generate and run on synthetic data
# test some parameters
import torch
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from modules.data_generation import generate_data
from modules.helpers import *
from models.classification import *

# SETUP ########################################################################
# load parameters
with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
print('')

# CUDA setup
cuda = torch.cuda.is_available()
print('GPU is available: ', cuda)
if cuda:
    torch.cuda.set_device(cfg['cuda_setup']['device_id'])

# model parameters
nn_params = cfg['model_params']
nn_params['cuda'] = cuda

# preprocessing
pre_process_paras = cfg['pre_process_paras']

# data generation parameters
data_gen_params = cfg['data_gen_params']

# classification
# kernel approximation (allows nonlinear, probabilistic multiclass but a lot quicker)
# samples n_comp samples to contruct RBF kernel (default gamma)
le = LabelEncoder()
def make_clf():
    nys = Nystroem(kernel='rbf', n_components=100)
    # logreg w/ default C
    lr = LogisticRegression(class_weight='balanced', C=1., penalty='l2', multi_class='multinomial', max_iter=1000)
    # stitch together
    clf = Pipeline([('nys',nys),('clf',lr)])

    return clf
clf = make_clf()

# output
outDir = 'synthetic_data'
################################################################################

### PARAMETER SEARCH ###########################################################
# parameters to test
param_dict = {'code_dim':[2,3,5,10],
              'base_lr':[0.01, 0.001,0.005,0.0001]}
#param_dict = {'base_lr':[0.01, 0.001,0.005,0.0001]}

if __name__ == '__main__':

    print('generating data')
    # generate data with different noise levels
    data_gen_params['set_noise'] = [0.15, 0.25, 0.33, 0.5]
    metadata, data = generate_data(n_subjects = data_gen_params['number_of_subjects'],
                                   n_tissue_types = data_gen_params['number_of_tissues'],
                                   n_voxels = data_gen_params['number_of_voxels'],
                                   n_features = data_gen_params['number_of_features'],
                                   noise = data_gen_params['set_noise'])

    train_idx, test_idx = train_test_split(np.unique(metadata['subjects']), test_size=0.2, shuffle=True, random_state=42)
    y_train, y_test = metadata[metadata['subjects'].isin(train_idx)], metadata[metadata['subjects'].isin(test_idx)]

    # MODEL TRAINING ########################################################################
    # for all parameter options
    palettes = ['Reds', 'Blues']
    for p, parameter_set in enumerate([*param_dict.keys()]):

        parameter_choices = [*param_dict.values()][p]

        # set up for results
        noise_level = []
        param_level = []
        recon_training = []
        recon_testing = []
        accuracy_training = []
        accuracy_testing = []
        log_loss_training = []
        log_loss_testing = []

        # for each noise level
        for nd, d in enumerate(data):
            print('')
            print('noise: {:.3f}'.format(data_gen_params['set_noise'][nd]))

            # DATA PREP #############################################################################
            # split into train and test
            x_train, x_test = d.loc[metadata['subjects'].isin(train_idx),:], d.loc[metadata['subjects'].isin(test_idx),:]

            # scale/standardise
            x_train, x_test = pre_process_datasets(x_train, x_test, pre_process_paras)

            # get cluster pairs, set up X data for BERMUDA
            dataset_list, _, cluster_pairs = prepare_data(x_train, y_train)
            test_data_set, _, _ = prepare_data(x_test, y_test)

            # for each parameter
            for npar, param in enumerate(parameter_choices):

                # reset params
                nn_params = cfg['model_params']
                nn_params['cuda'] = cuda
                nn_params['num_inputs'] = np.shape(dataset_list[0]['data'])[1]
                nn_params['log_interval'] = 100
                # set new param
                nn_params[parameter_set] = param

                # train and test
                train_recon, train_code, test_recon, test_code = train_and_test(dataset_list, y_train, cluster_pairs,
                                                                                  test_data_set, y_test,
                                                                                  nn_params)
                # recon error
                recon_training.append(mean_squared_error(x_train, train_recon))
                recon_testing.append(mean_squared_error(x_test, test_recon))

                # classification
                trained_model = train_classifier(train_code, le.fit_transform(y_train.tissues), clf)

                # predict
                train_predicted_proba = trained_model.predict_proba(train_code)
                test_predicted_proba = trained_model.predict_proba(test_code)

                # get accuracies
                ll_train, acc_train, _ = calculate_model_accuracy(le.transform(y_train.tissues), train_predicted_proba)
                log_loss_training.append(ll_train)
                accuracy_training.append(acc_train)

                ll_test, acc_test, _  = calculate_model_accuracy(le.transform(y_test.tissues), test_predicted_proba)
                log_loss_testing.append(ll_test)
                accuracy_testing.append(acc_test)

                # records params
                noise_level.append(data_gen_params['set_noise'][nd])
                param_level.append(param)

                print('')
                print('***interim results***')
                print(parameter_set)
                print('value: {:.4f}'.format(param))
                print('training data:')
                print('accuracy: {:.3f} log loss: {:.3f}'.format(accuracy_training[npar], log_loss_training[npar]))
                print('')
                print('test data')
                print('accuracy: {:.3f} log loss: {:.3f}'.format(accuracy_testing[npar], log_loss_testing[npar]))
                print('*********************')
                print('')

        # save out
        results = pd.DataFrame(np.vstack((noise_level, param_level, recon_training, recon_testing,
                                          log_loss_training, log_loss_testing,
                                          accuracy_training, accuracy_testing)).T)
        results.columns = ['noise_level', parameter_set, 'MSE_training', 'MSE_testing', 'log_loss_training', 'log_loss_testing', 'accuracy_training', 'accuracy_testing']
        results.to_csv(outDir + '/parameter_choice_' + str(parameter_set) + '_model_results.csv')

        plot_parameter_accuracies(results, parameter='MSE', palette=palettes[p], savepath=outDir + '/' + parameter_set + '-MSE.png')
        plot_parameter_accuracies(results, parameter='log_loss', palette=palettes[p], savepath=outDir + '/' + parameter_set + '-log_loss.png')
        plot_parameter_accuracies(results, parameter='accuracy', palette=palettes[p], savepath=outDir + '/' + parameter_set + '-accuracy.png')
##############################################################################################################################################
