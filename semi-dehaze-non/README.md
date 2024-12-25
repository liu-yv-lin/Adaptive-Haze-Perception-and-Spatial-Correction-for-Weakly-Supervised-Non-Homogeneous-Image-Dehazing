# Adaptive Haze Perception and Spatial Correction for Weakly Supervised Non-Homogeneous Image Dehazing

Authors: Aiping Yang, Yulin Liu, Yumeng Liu, Haoyang Bi, Liping Liu


### Abstract

This paper proposes a novel weakly supervised network for non-homogeneous image dehazing, comprising an Adaptive Haze Perception Module (AHPM) and a Spatial Information Correction Module (SICM). The AHPM employs a Dual Weighted Attention mechanism to dynamically focus on features within different channels and spatial locations, enabling the model to distinguish regions with varying haze density and distribution. This significantly enhances the perception of non-homogeneous haze. The SICM addresses spatial information loss by comparing down-sampled features with non-uniform haze features, feedbacking differences to correct structural deformations and detail blurriness. Extensive experiments demonstrate that our method achieves competitive performance compared to state-of-the-art weakly supervised dehazing methods and, in some cases, even supervised methods on benchmark datasets. By adaptively adjusting the dehazing strength based on haze concentration, our approach generates dehazed images with fine details and inherent colors, showcasing its effectiveness in real-world applications.


## O-HAZE Dataset

- Our O-HAZE dataset is available at:
https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/

## HAZERD Dataset

- Our HAZERD dataset is available at:https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/


## Prerequisites
Python 3.6 or above.


### Getting started


- Install PyTorch 1.6 or above and other dependencies (e.g., torchvision, visdom, tqdm, Pillow).
- Generate training and testing images:
```bash
python new_data_test.py 
python new_data_train.py 
```
### Training and Test

- Train the  model:
```bash
python train.py 
```
The checkpoints will be stored at `./experiment`.

- Test the model:
```bash
python test.py
```
The test results will be saved here: `./experiment`.



### Results
Performance comparison with the state-of-the-art approaches on O-HAZE and HAZERD Dataset  



   | Dataset  | O-HAZE |O-HAZE| HAZERD  |HAZERD |
   |-------|-------|-------|-------| ------------------------ |
   | Method  | PSNR  | SSIM  |  PSNR  | SSIM  |
   | SRT-SS-DAID     | 18.01 | 0.741 | 19.50 | 0.660 |
   | SDA-GAN | 17.91 | 0.726 | 18.20 | 0.795 |
   | SCA-Net | 18.13 | 0.724 |19.52 | 0.649 |
   | Ours    | 18.22 | 0.732 | 19.63 | 0.855 |



Visualization

Comparison between our method and other methods on O-HAZE dataset

![image](experiment/O-Haze_compare.png#pic_center)

Comparison between our method and other methods on real-world

![image](experiment/real world_compare.png#pic_center)




