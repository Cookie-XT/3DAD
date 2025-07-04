# Within 3DMM Space: Exploring Inherent 3D Artifact for Video Forgery Detection
## Abstract
 Recently, the breathtaking development and potential misuse of deepfake technology has raised numerous privacy and security concerns, triggering widespread apprehension. Existing deepfake detection methods focus on the analysis of local regions for faces, such as mouth movement, eye blinking frequency, etc., which, however, are limited in their ability to capture the global inconsistencies present in forged faces. Some researchers attempt to seize 3D artifacts related to facial global information, but typically treat the 3D information as mere input, lacking the in-depth analysis. To address these shortcomings and mine the inherent and delicate 3D artifacts in the forged faces, this paper innovatively proposes the 3D Artifact Detector (3DAD) method, which leverages the spatio-temporal inconsistency on the 3D semantic space in the forgery videos to uncover the deepfake clues. Specifically, we first employ 3D Analysis Unit (3DAU) to pre-train the face reconstruction task within 3D Morphable Model (3DMM) space, thereby obtaining the high-level inherent 3d representation. Concurrently, for the multi-levels of information in the face, we utilize the Texture Perception Unit (TPU) to extract the texture information in the low-level semantic space of the images. Ultimately we feed the two distinct modalities into the spatiotemporal fusion model for final detection. Through extensive intra- and cross-dataset experiments on publicly available datasets, we demonstrate the effectiveness and generalizability of the proposed method.  The source code is available at <https://github.com/Cookie-XT/3DAD>. 
## Frameworrk
![image](https://github.com/user-attachments/assets/1ebe6a05-c156-4951-add4-a6dba073b466)
## Installation
### Requirement
We test the code on python==3.10 and pytorch==1.12.1
### Dataset
1.In our experiment we use FaceForensics++, WildDeepfake, CelebDF datasets for evaluation.\
2.Please divide the video into groups of 32 frames each
## Usage
### train
First, you should download the model of 3d and texture, and you can download the "pretrain_model" from Google Drive link <https://drive.google.com/drive/folders/1WAcSLOi_d_ub2Z0MfqoEO9LNpUAEi56x?usp=drive_link>. and update the corresponding model file paths.

`python train.py --outf="your output path"`
### test
Your can download our model from Google Drive link <https://drive.google.com/drive/folders/1WAcSLOi_d_ub2Z0MfqoEO9LNpUAEi56x?usp=drive_link>  based from FF++,Celeb-df and WildDeepfake.\
`python test.py`
## Acknowledge
1.Towards Fast, Accurate and Stable 3D Dense Face Alignmen. [[code](https://github.com/cleardusk/3DDFA_V2)] \
2.Res2net: A new multi-scale backbone architecture. [[code](https://github.com/Res2Net)]\
3.Pyramid Spatial-Temporal Aggregation for Video-based Person Re-Identification [[code](https://github.com/WangYQ9/VideoReID-PSTA)]
