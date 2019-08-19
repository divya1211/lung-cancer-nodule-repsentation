# Lung Cancer Nodule Repsentation

##TO-Do
-3d neural network switch 
-confirm accuracy
-area under precission recall curve
-extract more data
-other things you can predict(lobulation sphericity)
-build smaller network 



## Introduction

Representation Learning through medical imaging of chest
Cancer treatment often needs deals with evaluating  chance of survival for patients which depends on a multitude of factors. Statistics help determine which treatment would be best and help doctors weight risks against benefits.  
This research would involve using representation learning for Predicting the survivability of lung cancer patients. We would be using a Lung Image Database Consortium image collection (LIDC-IDRI) for 
 learning and finding a best representation of lung cancer from the dataset. This dataset contains nodules with segmentation and readings from Radiologists. Representation learned are tested further on a smaller dataset of 415 patients containing survival outcome of patients and disease stage. Primary goal of Survival prediction depends on two latent factors, Aggressiveness and fitness score which are the underlying biological factors for survival prediction. The sub tasks would be learning optimal representations for calculating aggressiveness and fitness scores. Also we would be trying to predict other confounders such as age, stage and gender using the learned representations

## Goals

To generate a good representation of the nodule using unsupervised learning

## TASKS done so far

- the dataset is  collected from Cancer Imaging archive https://www.cancerimagingarchive.net/. Generally the medical images are in a NIFTI format. The  normal format of a CT scan for a patient consists of multiple slices of images and a Xml file. The 2-dimensional images of a "slice" or section of the body consists of data that can also be used to construct 3-dimensional images which can be rotated and viewed in any orientation to provide a more natural and functional view of the patient's anatomy.

- The dataset is converted firstly from Nifti to nrrd format.
- Cropping nodule volumes of size 64*64*64 from the entire volume of no of slices*512*512. As the task at hand is to generate a good representation of the nodule hence we need the model to not focus on learning the representation of the area around ie the lung
- The assumption of extracting a cube of 64*64*64 has been drawn after calculating the maximum diameter of the nodules (511).

- Designing the Data loader:- using the medical torch data loader as a guideline we create a data loader to load the tensors(after converting the nrrd to tensors)

- Extracting labels:- from the xml file for malignancy of Nodule 1 for 511 patients
- A class named Data is created which returns a pair of tensor and a label corresponding to the particular nodule(malignancy.pt) whereas the tensor is loaded from nodule.pt.

## Supervised learning

To get an idea of how well we could perform under a supervised training setup. We use a pretrained resnet18. And after fine-tuning it's fully connected layer for final number of features to 5 as we have 5 classes of malignancies.
Important tweaks:-

- we have skipped 3 patients namely 185,157,212 as they have a glich in data. The no. of slices for a patient is less than the number of slices for a given annotation.
- to pass through a reset18 we need to compress the 3d data to (3, n, n). Hence the 3d tensor of size 64*64*64 has been compressed after taking mean along each dimension.




## Unsupervised Tasks

- AMDIM: - Maximizing mutual information between features which have been extracted from multiple views of a shared context
	 positive samples and negative samples are passed through an encoder which learns to identify or discriminate between the ground truth and false.
	 positive sample is the ground truth whereas the negative samples are drawn from any other image from the batch size except the positive image. Both images are passed through a series of 		 tranformations. In our modified adaption of the paper the images have only been transposed or rotated along random axises as all other transformations would not be so efficient and would destroy 	 the nodule representation. The accuracy obtained is 31%. The experiments were run for 49 epochs. Using just the nodules as training images instead of the entire CT scan helps reduce noise which 	 Is also learnt in the process of learning a good representation as shown by the paper DIM.


## Motivation for Unsupervised Learning

The task of generating such a problem generator such that the model tries to capture useful information about the data in order to solve the generated problem.

AMDIM also introduced a concept of shared context which can usefully be exploited to induce more tasks around unsupervised learning. The important takeaway is trying to preserve this mutual information or shared context. As it has been established that this shared context helps preserve high level features whose influence is over multiple views.

Tasks
If we imagine the nodule to be a cube of dimension 64*64*64 and task at hand being to produce a good representation of this cube. If this cube were to be sliced into smaller cubes or voxels. Then the task gets subdivided into learning a good representation of each voxel.  


![](https://github.com/divya1211/lung-cancer-nodule-repsentation/blob/master/diag.png)
