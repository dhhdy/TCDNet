# TCDNet: Double-Branch Feature Fusion for Lung Nodule Segmentation via Combining Transformer and CNN

.<div align=center><img src="https://github.com/user-attachments/assets/64fcdb75-8419-45c9-8444-b4a655419652" width="700" height="450" /></div>

# Overview
We propose a novel network structure named TCDNet, which adopts Transformer and CNN as its Dual-branch feature extractor to progressively sample the multi-scale channel semantic information while capturing both global and local information. Two innovative modules named Shallow Fusion (SF) and Feature Aggregation (FA) and an improved upsampling structure called Dense Connection Based on Element-wise Addition (DCEA) are proposed to address the problem caused by the imbalance feature weight information of the CT images. The proposed method achieves promising results in the LIDC-IDRI and in-house datasets.

# Setup
**1. Prerequisites**
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on single NVIDIA Tesla A40 GPU)
* CUDA + cuDNN (Tested on CUDA 11.8 and cuDNN 8.7)
* torch>=2.0.0, python==3.9

# Pretrained
the pretrained weight of the dual-branch extractor can be found here
EVA-02: https://github.com/baaivision/EVA/tree/master/EVA-02/asuka

ConvNext: https://github.com/facebookresearch/ConvNeXt

Thanks for their outstanding works!



