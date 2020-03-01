# BERMUDA: Batch Effect ReMoval Using Deep Autoencoders
Original authors:
Tongxin Wang, Travis S Johnson, Wei Shao, Zixiao Lu, Bryan R Helm, Jie Zhang and Kun Huang
Original code and data for using [BERMUDA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1764-6 "BERMUDA"), a novel transfer-learning-based method for batch-effect correction in single cell RNA sequencing (scRNA-seq) data.

## WIP: Application to paediatric brain tumour MRI data
Using BERMUDA to define a latent space capturing tissue-specific variation in diffusion MRI parameters, corrected for inter-subject variation (cf. batch).

## Dependencies
* Python 3.6.5
* scikit-learn 0.19.1
* pyTorch 0.4.0
* imbalanced-learn 0.3.3
* universal-divergence 0.2.0

## Files
*main_pancreas.py*: An Example of combining two pancreas datasets\
*main_pbmc.py*: An Example of combining PBMCs with pan T cells\
*R/pre_processing.R*: Workflow of detecting clusters using Seurat and identifying similar clusters using MetaNeighbor\
*R/gaussian.R*: Simulate data based on 2D Gaussian distributions\
*R/splatter.R*: Simulate data using Splatter package

## Cite
Wang, T., Johnson, T.S., Shao, W. et al. BERMUDA: a novel deep transfer learning method for single-cell RNA sequencing batch correction reveals hidden high-resolution cellular subtypes. Genome Biol 20, 165 (2019) doi:10.1186/s13059-019-1764-6
