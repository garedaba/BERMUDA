# BERMUDA: Batch Effect ReMoval Using Deep Autoencoders
Original authors:
Tongxin Wang, Travis S Johnson, Wei Shao, Zixiao Lu, Bryan R Helm, Jie Zhang and Kun Huang
Original code and data for using [BERMUDA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1764-6 "BERMUDA"), a novel transfer-learning-based method for batch-effect correction in single cell RNA sequencing (scRNA-seq) data.

## WIP: Application to paediatric brain tumour MRI data
Using BERMUDA to define a latent space capturing tissue-specific variation in diffusion MRI parameters, corrected for inter-subject variation (cf. batch).

## Dependencies
* Python 3.7.6
* scikit-learn 0.22.1
* pyTorch 1.4.0
* rpy2 2.9.4
* R libraries: simstudy, clusterGeneration, Matrix

## Files
*main_synthetic.py*: example using synthetic data

### todo
* ~change data generation to vary number of voxels in each tissue~
* ~improve plots~
* ~classifier in latent space, project to discriminative axes?~
* plots for proba classifications
* implement homogeneity/divergence scores for train/test data?


## Cite
Wang, T., Johnson, T.S., Shao, W. et al. BERMUDA: a novel deep transfer learning method for single-cell RNA sequencing batch correction reveals hidden high-resolution cellular subtypes. Genome Biol 20, 165 (2019) doi:10.1186/s13059-019-1764-6
