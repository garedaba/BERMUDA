import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from imblearn.over_sampling import RandomOverSampler
# imblearn_seed = 0

import math
import time

import numpy as np

import models.autoencoder as models
from models.mmd import mix_rbf_mmd2

# range of sigma for MMD loss calculation
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

def safe_div(n, d):
    return n / d if d else n

def training(dataset_list, cluster_pairs, nn_paras):
    """ Training an autoencoder to remove batch effects
    dataset_list: list of datasets for batch correction
    cluster_pairs: pairs of similar clusters with weights
    nn_paras: parameters for neural network training

    returns:
    model: trained autoencoder
    loss_total_list: list of total loss
    loss_reconstruct_list: list of reconstruction loss
    loss_transfer_list: list of transfer loss
    """
    # load nn parameters
    batch_size = nn_paras['batch_size']
    num_epochs = nn_paras['num_epochs']
    num_inputs = nn_paras['num_inputs']
    code_dim = nn_paras['code_dim']
    cuda = nn_paras['cuda']
    layers = nn_paras['layers']

    # create data loaders - one per tissue type
    cluster_loader_dict = {}
    for i in range(len(dataset_list)):
        data = dataset_list[i]['data']
        cluster_labels = dataset_list[i]['tissue_labels']
        unique_labels = np.unique(cluster_labels)
        # oversample small clusters
        data, cluster_labels = RandomOverSampler().fit_sample(data, cluster_labels)

        # construct DataLoader list
        for j in range(len(unique_labels)):
            idx = cluster_labels == unique_labels[j]

            if cuda:
                torch_dataset = torch.utils.data.TensorDataset(
                                       torch.FloatTensor(data[idx,:]).cuda(), torch.LongTensor(cluster_labels[idx]).cuda())
            else:
                torch_dataset = torch.utils.data.TensorDataset(
                                        torch.FloatTensor(data[idx, :]), torch.LongTensor(cluster_labels[idx]))

            data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=min(len(data[idx,:]),batch_size),
                                                      shuffle=True, drop_last=False)

            cluster_loader_dict[unique_labels[j]] = data_loader
            # each dict element (one per tissue type) contains data loader with batches of all examples of that tissue type across subjects

    # create model
    if layers==1:
        model = models.autoencoder_2(num_inputs=num_inputs, code_dim=code_dim)
    else:
        model = models.autoencoder_3(num_inputs=num_inputs, code_dim=code_dim)

    if cuda:
        model.cuda()

    # model training
    loss_total_list = []  # list of total loss
    loss_reconstruct_list = []
    loss_transfer_list = []

    print('{:} MODEL TRAINING'.format(time.asctime(time.localtime())))

    for epoch in range(1, num_epochs + 1):
        avg_loss, avg_reco_loss, avg_tran_loss = training_epoch(epoch, model, cluster_loader_dict, cluster_pairs, nn_paras)
        # terminate early if loss is nan
        if math.isnan(avg_reco_loss) or math.isnan(avg_tran_loss):
            return [], model, [], [], []
        loss_total_list.append(avg_loss)
        loss_reconstruct_list.append(avg_reco_loss)
        loss_transfer_list.append(avg_tran_loss)

    return model, loss_total_list, loss_reconstruct_list, loss_transfer_list


def training_epoch(epoch, model, cluster_loader_dict, cluster_pairs, nn_paras):
    """ Training a single epoch
    epoch: number of the current epoch
    model: autoencoder
    cluster_loader_dict: dict of DataLoaders indexed by clusters
    cluster_pairs: pairs of similar clusters with weights
    nn_paras: parameters for neural network training

    returns:
    avg_total_loss: average total loss of mini-batches
    avg_reco_loss: average reconstruction loss of mini-batches
    avg_tran_loss: average transfer loss of mini-batches
        """

    log_interval = nn_paras['log_interval']
    # load nn parameters
    base_lr = nn_paras['base_lr']
    lr_step = nn_paras['lr_step']
    num_epochs = nn_paras['num_epochs']
    l2_decay = nn_paras['l2_decay']
    gamma = nn_paras['gamma']
    cuda = nn_paras['cuda']

    # step decay of learning rate
    learning_rate = base_lr / math.pow(2, math.floor(epoch / lr_step))

    # regularization parameter between two losses, increasing over time
    #gamma_rate = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1
    gamma_rate = 2 / (1 + math.exp(-5 * (epoch) / num_epochs)) - 1
    gamma = gamma_rate * gamma

    if epoch % log_interval == 0:
        print('epoch {}\t learning rate {:.4f}\t gamma {:.4f}\t'.format(epoch, learning_rate, gamma))

    optimizer = torch.optim.Adam([{'params': model.encoder.parameters()},
                                  {'params': model.decoder.parameters()}],
                                 lr=learning_rate, weight_decay=l2_decay)

    model.train()

    iter_data_dict = {}
    # for each tissue type
    for cls in cluster_loader_dict:
        iter_data = iter(cluster_loader_dict[cls])
        iter_data_dict[cls] = iter_data
    # use the largest dataset to define an epoch
    num_iter = 0
    for cls in cluster_loader_dict:
        num_iter = max(num_iter, len(cluster_loader_dict[cls]))
        # largest number of batches within a tissue type (subjects combined)

    total_loss = 0
    total_reco_loss = 0
    total_tran_loss = 0
    num_batches = 0

    # for each batch
    for it in range(0, num_iter):
        data_dict = {}
        label_dict = {}
        code_dict = {}
        reconstruct_dict = {}
        # take one batch per cluster
        for cls in iter_data_dict:
            data, labels = iter_data_dict[cls].next() # next batch
            data_dict[cls] = data
            label_dict[cls] = labels
            if it % len(cluster_loader_dict[cls]) == 0:
                iter_data_dict[cls] = iter(cluster_loader_dict[cls]) # if ran out, start again
            data_dict[cls] = Variable(data_dict[cls]) # one batch per element
            label_dict[cls] = Variable(label_dict[cls])

        for cls in data_dict: # for each batch
            code, reconstruct = model(data_dict[cls]) # train
            code_dict[cls] = code
            reconstruct_dict[cls] = reconstruct
        # model trained with all batches, one per cluster
        optimizer.zero_grad()

        # transfer loss for cluster pairs in cluster_pairs matrix
        loss_transfer = torch.FloatTensor([0])
        if cuda:
            loss_transfer = loss_transfer.cuda()
        for i in range(cluster_pairs.shape[0]):
            cls_1 = int(cluster_pairs[i,0])
            cls_2 = int(cluster_pairs[i,1])
            if cls_1 not in code_dict or cls_2 not in code_dict:
                continue
            mmd2_D = mix_rbf_mmd2(code_dict[cls_1], code_dict[cls_2], sigma_list)
            loss_transfer += mmd2_D * cluster_pairs[i,2]

        # reconstruction loss for all clusters
        loss_reconstruct = torch.FloatTensor([0])
        if cuda:
            loss_reconstruct = loss_reconstruct.cuda()
        for cls in data_dict:
            loss_reconstruct += F.mse_loss(reconstruct_dict[cls], data_dict[cls])

        # ortho loss
        #ortho_loss = torch.FloatTensor([0])
        #if cuda:
        #    ortho_loss = ortho_loss.cuda()
        #for cls in code_dict:
        #    ortho_loss += F.mse_loss(torch.matmul(torch.transpose(code_dict[cls],1,0), code_dict[cls]), torch.eye(code_dict[cls].shape[1]).cuda())

        #loss = (loss_reconstruct + (0.1*ortho_loss)) + gamma * loss_transfer
        loss = (loss_reconstruct) + gamma * loss_transfer

        loss.backward()
        optimizer.step()

        # update total loss
        num_batches += 1
        total_loss += loss.data.item()
        total_reco_loss += loss_reconstruct.data.item()
        total_tran_loss += loss_transfer.data.item()

    avg_total_loss = safe_div(total_loss, num_batches)
    avg_reco_loss = safe_div(total_reco_loss, num_batches)
    avg_tran_loss = safe_div(total_tran_loss, num_batches)

    if epoch % log_interval == 0:
        print('average_loss {:.3f}\t average_reconstruct_loss {:.3f}\t average_transfer_loss {:.3f}'.format(
            avg_total_loss, avg_reco_loss, avg_tran_loss))
    return avg_total_loss, avg_reco_loss, avg_tran_loss


def testing(model, dataset_list, nn_paras):
    """ apply trained model to data
    model: trained autoencoder
    dataset_list: list of datasets to model
    nn_paras: parameters for neural network training

    code_list: list of embedded codes
    recon_list: reconstructed data
    """

    # load nn parameters
    cuda = nn_paras['cuda']

    data_loader_list = []
    num_cells = []
    for dataset in dataset_list:

        torch_dataset = torch.utils.data.TensorDataset(
                                         torch.FloatTensor(dataset['data']), torch.LongTensor(dataset['sample_labels']))

        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=len(dataset['sample_labels']),
                                                    shuffle=False)
        data_loader_list.append(data_loader)
        num_cells.append(len(dataset["sample_labels"]))

    model.eval()

    code_list = [] # list of embedded codes
    recon_list = []

    for i in range(len(data_loader_list)):
        idx = 0
        with torch.no_grad():
            for data, labels in data_loader_list[i]:
                if cuda:
                    data, labels = data.cuda(), labels.cuda()
                code_tmp, recon_tmp = model(data)
                code_tmp = code_tmp.cpu().numpy()
                recon_tmp = recon_tmp.cpu().numpy()

                if idx == 0:
                    code = np.zeros((code_tmp.shape[1], num_cells[i]))
                    recon = np.zeros((recon_tmp.shape[1], num_cells[i]))

                code[:, idx:idx + code_tmp.shape[0]] = code_tmp.T
                recon[:, idx:idx + recon_tmp.shape[0]] = recon_tmp.T
                idx += code_tmp.shape[0]

        code_list.append(code)
        recon_list.append(recon)

    return code_list, recon_list
