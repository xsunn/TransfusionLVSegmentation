# Transformer Based Feature Fusion for Left Ventricle Segmentation in 4D Flow MRI

## 1. Introduction
The repository is the official code based on Pytorch for the paper of <br>
``
"Transformer Based Feature Fusion for Left Ventricle Segmentation in 4D Flow MRI"  MICCAI 2022, Singapore
``
https://link.springer.com/chapter/10.1007/978-3-031-16443-9_36 <br>
Four-dimensional flow magnetic resonance imaging (4D Flow MRI) enables visualization of intra-cardiac blood flow and quantification of cardiac function using time-resolved three directional velocity data. Segmentation of cardiac 4D flow data is a big challenge due to the extremely poor contrast between the blood pool and myocardium. The magnitude and velocity images from a 4D flow acquisition provide complementary information, but how to extract and fuse these features efficiently is unknown. Automated cardiac segmentation methods from 4D flow MRI have not been fully investigated yet. In this paper, we take the velocity and magnitude image as the inputs of two branches separately, then propose a Transformer based cross- and self-fusion layer to explore the inter-relationship from two modalities and model the intra-relationship in the same modality. A large in-house dataset of 104 subjects (91,182 2D images) was used to train and evaluate our model using several metrics including the Dice, Average Surface Distance (ASD), end-diastolic volume (EDV), end-systolic volume (ESV), Left Ventricle Ejection Fraction (LVEF) and Kinetic Energy (KE). Our method achieved a mean Dice of 86.52%, and ASD of 2.51 mm. Evaluation on the clinical parameters demonstrated competitive results, yielding a Pearson correlation coefficient of 83.26%, 97.4%, 96.97% and 98.92% for LVEF, EDV, ESV and KE respectively.

## 2. Training data structure

The data structure is as following:<br>

Data <br>
└── Patient1 <br>
&emsp;&emsp;├── SAX4DFMAG <br> 
&emsp;&emsp;├── SAX4DFX <br>
&emsp;&emsp;├── SAX4DFY <br>
&emsp;&emsp;├── SAX4DFZ <br>
└── Patient2 <br>
&emsp;&emsp;├── SAX4DFMAG <br>
&emsp;&emsp;├── SAX4DFX <br>
&emsp;&emsp;├── SAX4DFY <br>
&emsp;&emsp;├── SAX4DFZ <br>


The training data path is saved in TXT file as following:<br>
.../DATA/Patient1/SAX4DFMASK/IM_sl0010_ph0001.dcm <br>
.../DATA/Patient1/SAX4DFMASK/IM_sl0010_ph0002.dcm <br>
...<br>
...<br>
...<br>

## 3. Model structure
### Feature fusion layer
![image text](https://github.com/xsunn/4DFlowLVSeg/blob/main/ModelStructure/SAL.png)
### Segmentation network
![image text](https://github.com/xsunn/4DFlowLVSeg/blob/main/ModelStructure/UnetFeatureFusion.png)
