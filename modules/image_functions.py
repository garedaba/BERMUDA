import sys, os, glob, subprocess
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from scipy.stats.mstats import winsorize

from nilearn.input_data import NiftiMasker
#from PIL import Image
#from sklearn.preprocessing import StandardScaler

def preprocess_subject(subject, datapath):
    """create lesion and healthy tissue masks from subject data
    mainly using clunky bash calls

    requires:
    - processed diffusion data niftis
    - co-registered GM, WM, CSF prob. segmentation maps
     - [gm/wm/csf]_resample_masked.nii.gz
    - hand-labeled lesion maps in diffusion space
     - ls_*.nii.gz
     """
    print("")
    print("********PRE-PROCESSING IMAGING DATA....************")
    print("")
    subject_dir = (datapath + subject)

    # create 4D diffusion file
    if os.path.isfile((subject_dir + '/all_diffusion.nii.gz')) is False:
        bashCommand = ("mrcat "
                          + subject_dir + "/b1000_fa.nii.gz " \
                          + subject_dir + "/b1000_adc.nii.gz " \
                          + subject_dir + "/ShZero_tournier.nii.gz " \
                          + subject_dir + "/After_Noddi_ficvf_up.nii.gz " \
                          + subject_dir + "/After_Noddi_fiso_up.nii.gz " \
                          + subject_dir + "/After_Noddi_odi_up.nii.gz " \
                          + subject_dir + "/SMT_rician_diff_up.nii.gz " \
                          + subject_dir + "/SMT_rician_extramd_up.nii.gz " \
                          + subject_dir + "/SMT_rician_extratrans_up.nii.gz " \
                          + subject_dir + "/SMT_rician_intra_up.nii.gz " \
                          + subject_dir + "/all_diffusion.nii.gz ")
        subprocess.check_call(['bash','-c', bashCommand])
        # mask with FA image
        bashCommand = ("mrthreshold  -abs 1e-8 " + subject_dir + "/b1000_fa.nii.gz - | mrcalc -force - " + subject_dir + "/all_diffusion.nii.gz  " \
                       " -mult " + subject_dir + "/all_diffusion.nii.gz ")
        subprocess.check_output(['bash','-c', bashCommand])

    # create lesion mask from all classes
    if os.path.isfile((subject_dir + '/all_lesion_inverse.nii.gz')) is False:
        all_ls = glob.glob(subject_dir + '/ls*')
        if len(all_ls)==1:
            bashCommand = ("mrthreshold -abs 1e-8 " + subject_dir + "/b1000_fa.nii.gz  - | "\
                           "mrcalc -force - " + all_ls[0] + " -mult " + subject_dir + "/all_lesion.nii.gz; "\
                           "mrthreshold -abs 1e-8 -invert " + subject_dir + "/all_lesion.nii.gz " + subject_dir + "/all_lesion_inverse.nii.gz")
        else:
            bashCommand = ("mrmath " + subject_dir + "/ls_* max " + subject_dir + "/all_lesion.nii.gz; "\
                           "mrthreshold -abs 1e-8 " + subject_dir + "/b1000_fa.nii.gz - | "\
                           "mrcalc -force - " + subject_dir + "/all_lesion.nii.gz -mult " + subject_dir + "/all_lesion.nii.gz; "\
                           "mrthreshold -abs 1e-8 -invert " + subject_dir + "/all_lesion.nii.gz " + subject_dir + "/all_lesion_inverse.nii.gz")
        subprocess.check_output(['bash','-c', bashCommand])


    for n,tis in enumerate(['gm', 'wm', 'csf']):
        if os.path.isfile((subject_dir + "/" + tis + "_resample_masked.nii.gz")) is False:
            print("extracting initial tissue mask for " + tis)
            bashCommand = ( "mrthreshold -abs .8 " + subject_dir + "/" + tis + "_resample.nii.gz - | "\
                            "maskfilter - erode - | "\
                            "mrcalc - " + subject_dir + "/all_lesion_inverse.nii.gz -mult - | "\
                            "mrcalc  - " + subject_dir + "/b1000_fa.nii.gz -mult - | "\
                            "mrthreshold -abs 1e-8 -force - " + subject_dir + "/" + tis + "_resample_masked.nii.gz" )
            subprocess.check_output(['bash','-c', bashCommand])

    print("")
    print("********PRE-PROCESS COMPLETE....************")
    print("")

import sys, os, glob, subprocess
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from nilearn.input_data import NiftiMasker

def extract_lesion_voxels(subject_dir, lesion_class, num_vox=500, winsorise=True, train_test=True):
    num_vox=num_vox
    lesion_file = (subject_dir + "/" + lesion_class + "_final_resample.nii.gz")

    # load diffusion data within lesion mask
    mask = NiftiMasker(mask_img=lesion_file)
    diffusion_data = mask.fit_transform(subject_dir + '/all_diffusion.nii.gz')
    diffusion_data = diffusion_data[np.sum(diffusion_data,1)>0,:] # ensure non-zero

    # update num_vox if there are fewer than 500 voxels in mask
    if np.shape(diffusion_data)[1]<num_vox:
        num_vox=np.shape(diffusion_data)[1]

    print(lesion_class)
    if train_test is True:
        print(str(num_vox) + " voxels for both training and testing")
    else:
        print(str(num_vox) + " voxels for training only")

    # space for data
    lesion_class_diffusion_training = np.zeros((num_vox, 10))
    if train_test is True:
        lesion_class_diffusion_testing = np.zeros((num_vox, 10))

    # space for sampled voxel indices
    voxel_ind_training = np.zeros((num_vox,))
    if train_test is True:
        voxel_ind_testing = np.zeros((num_vox,))

    # shuffle and take top <num_vox>
    rand_ind = np.arange(np.shape(diffusion_data)[1])
    np.random.shuffle(rand_ind)
    rand_ind_training = rand_ind[:num_vox]
    if train_test is True:
        rand_ind_testing = rand_ind[num_vox:num_vox*2] # take the next num_vox for testing

    sampled_data_training = diffusion_data[:,rand_ind_training]
    if train_test is True:
        sampled_data_testing = diffusion_data[:,rand_ind_testing]

    lesion_class_diffusion_training = sampled_data_training.T

    if train_test is True:
        lesion_class_diffusion_testing[low_ind:high_ind,:] = sampled_data_testing.T

    # remove top and bottom 5% outliers
    if winsorise==True:
        lesion_class_diffusion_training = winsorize(lesion_class_diffusion_training, limits=(0.05), axis=0).data

    voxel_ind_training = rand_ind_training
    if train_test is True:
        voxel_ind_testing = rand_ind_testing

    label = pd.DataFrame([lesion_class]*num_vox, columns=['tissues'])

    lesion_class_diffusion_training = pd.DataFrame(np.column_stack((voxel_ind_training, label, lesion_class_diffusion_training)))
    lesion_class_diffusion_training.rename(columns={0:'voxel', 1:'tissues', 2:'fa',3:'adc',4:'sh0',5:'ficvf',6:'fiso',7:'odi',8:'smt_diff',9:'smt_extramd',10:'smt_extratrans',11:'smt_intra'}, inplace=True)

    if train_test is True:
        lesion_class_diffusion_testing = pd.DataFrame(np.column_stack((voxel_ind_testing, label, lesion_class_diffusion_testing)))
        lesion_class_diffusion_testing.rename(columns={0:'voxel',1:'tissues', 2:'fa',3:'adc',4:'sh0',5:'ficvf',6:'fiso',7:'odi',8:'smt_diff',9:'smt_extramd',10:'smt_extratrans',11:'smt_intra'}, inplace=True)

    if train_test is True:
        return lesion_class_diffusion_training, lesion_class_diffusion_testing
    else:
        return lesion_class_diffusion_training

def extract_healthy_voxels(subject_dir, num_vox=500, winsorise=True, train_test=True):

    print("")
    print("********EXTRACTING DIFFUSION DATA....************")
    if train_test is True:
        print(str(num_vox) + " voxels for both training and testing")
    else:
        print(str(num_vox) + " voxels for training only")
    print("")

    # space for data
    tissue_class_diffusion_training = np.zeros((num_vox*3, 10))
    if train_test is True:
        tissue_class_diffusion_testing = np.zeros((num_vox*3, 10))
    # voxel indices
    voxel_ind_training = np.zeros((num_vox*3,))
    if train_test is True:
        voxel_ind_testing = np.zeros((num_vox*3,))

    for n,tis in enumerate(['cortical', 'wm', 'ventricle']):
        if os.path.isfile((subject_dir + "/" + tis + "-mask.nii.gz")) is False:
            print(tis + " mask not available  -  please check")

            return

        tissue_mask = NiftiMasker(mask_img=subject_dir + '/' + tis +"-mask.nii.gz")
        tissue_diffusion_data = tissue_mask.fit_transform(subject_dir + "/all_diffusion.nii.gz")
        tissue_diffusion_data = tissue_diffusion_data[abs(np.sum(tissue_diffusion_data,1))>0,:] # ensure non-zero

        # shuffle and take top <num_vox>
        rand_ind = np.arange(np.shape(tissue_diffusion_data)[1])
        np.random.shuffle(rand_ind)
        rand_ind_training = rand_ind[:num_vox]
        if train_test is True:
            rand_ind_testing = rand_ind[num_vox:num_vox*2]

        sampled_data_training = tissue_diffusion_data[:,rand_ind_training].T
        if train_test is True:
            sampled_data_testing = tissue_diffusion_data[:,rand_ind_testing].T

        low_ind = num_vox * n
        high_ind = num_vox * (n+1)

        if winsorise==True:
            sampled_data_training = winsorize(sampled_data_training, limits=(0.05), axis=0).data

        tissue_class_diffusion_training[low_ind:high_ind,:] = sampled_data_training

        if train_test is True:
            tissue_class_diffusion_testing[low_ind:high_ind,:] = sampled_data_testing
        voxel_ind_training[low_ind:high_ind,] = rand_ind_training
        if train_test is True:
            voxel_ind_testing[low_ind:high_ind,] = rand_ind_testing

        print(tis)

    label = pd.DataFrame(np.concatenate((np.array(['cortical']*num_vox), np.array(['wm']*num_vox), np.array(['csf']*num_vox)), axis=0), columns=['tissues'])

    tissue_class_diffusion_training = pd.DataFrame(np.column_stack((voxel_ind_training, label, tissue_class_diffusion_training)))
    tissue_class_diffusion_training.rename(columns={0:'voxel',1:'tissues',2:'fa',3:'adc',4:'sh0',5:'ficvf',6:'fiso',7:'odi',8:'smt_diff',9:'smt_extramd',10:'smt_extratrans',11:'smt_intra'}, inplace=True)

    if train_test is True:
        tissue_class_diffusion_testing = pd.DataFrame(np.column_stack((voxel_ind_testing, label, tissue_class_diffusion_testing)))
        tissue_class_diffusion_testing.rename(columns={0:'voxel',1:'tissues',2:'fa',3:'adc',4:'sh0',5:'ficvf',6:'fiso',7:'odi',8:'smt_diff',9:'smt_extramd',10:'smt_extratrans',11:'smt_intra'}, inplace=True)

    if train_test is True:
        return tissue_class_diffusion_training, tissue_class_diffusion_testing
    else:
        return tissue_class_diffusion_training

def get_image_data(subject):
    mask = NiftiMasker(mask_img='subject_data/' + subject + '/mask.nii.gz', mask_strategy='background')
    image_data = mask.fit_transform('subject_data/' + subject + '/all_diffusion.nii.gz')

    return image_data, mask

"""
def get_colours(x,y):
    imgobj = Image.open('modules/tissue_projection/zeigler.png')
    rgb_list = np.asarray(imgobj.getdata())
    rgb_image = np.reshape(rgb_list, [512,512,3])

    list_xy = (np.asarray([x, y])).T
    print("number of points: ", len(list_xy))

    min_x_data = min(list_xy[:,0])
    max_x_data = max(list_xy[:,0])
    min_y_data = min(list_xy[:,1])
    max_y_data = max(list_xy[:,1])

    rgb = np.zeros((len(list_xy),3))

    for i in np.arange(len(list_xy)):
        x = list_xy[i,0] - min_x_data
        y = list_xy[i,1] - min_y_data
        x = (float(x) / (max_x_data-min_x_data))
        y = (float(y) / (max_y_data-min_y_data))
        rgb[i,:] = rgb_image[int(np.round(511*x)), int(np.round(511*y)),:]

    return rgb
"""
