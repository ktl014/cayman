# README
Cayman - AlexNet Model trained for binary fish egg classification 

Overview
- Cayman is a collection of python scripts to train a CNN for binary fish egg classification, which serves the purpose for estimating Nassau Grouper population sizes.
- Version: 1.00a
- Authors: Kevin Le

# Prerequisites
- python 2.7. (with scikit-image and lmdb)
- caffe (with pycaffe)

## Tour
#### Code
1. model.py: Defines the Classification Model
2. train.py: Training script
3. eval.py: Evaluation script
4. tools/dataset.py: Script for preparing datasets
5. tools/create_lmdb.py: Script following dataset creation by converting to lmdbs
6. tools/circle_detection.py: Object size detection script
7. tools/visualize_db.ipynb: Data visualization script

#### Data
1. Training set: data/${VERSION}/data_train.csv
2. Validation set: data/${VERSION}/data_val.csv
3. Test (unlabelled) set: data/${VERSION}/data_test.csv

## Examples
To view the model perform image detection, the user can use the basic demo (HIGHLIGHT DEMO)
Run this ipython notebook

## Results
#### Binary Fish Egg Classification

SHOW GIF OF BINARY FISH EGG DETECTION

## Content
1. Introduction
2. Problem
3. Dataset & Preprocessing
4. Model
5. Object Size Detection
6. Conclusion

### Part 1: Introduction
What are Nassau Groupers?
Where are they found?
Why are they interesting?
What research has come out of this interest?

From this field study, the massive amount of data to process by hand leads us to our problem.

### Part 2: Problem
##### The "Fishy" Dilemna
Given 225,000 images collected from the Cayman Field study, how do we determine the Nassau population size from this distribution? 

##### Client
PhD Biologist Candidate from the Semmens Lab & Jaffe Lab

##### Objective
Develop a model to detect all possible fish eggs from the sample and measure the size of the predicted fish eggs to determine it as a Nassau species

### Part 3: Dataset & Preprocessing
##### Dataset
With the need for fish egg detection, the dataset will be organized for binary fish egg classification. Originallly, the data
was given with up to 18 classes labeled. To train our classifier to perform well for fish egg detection, we need to
sample a minimum amount of images from 17 of the 18 classes and categorize it as our non fish egg, while the remaining class is defined as the fish egg class.
Below are image representations of the 18 classes:

INSERT PICTURE HERE

After training the model and receiving reasonable performance on the validation set, the model will be deployed on the test set, which is the main objective of this project
in regards to detecting fish eggs. 

| Dataset      | # of Non Fish Egg Examples |# of Fish Egg Examples|Total Examples|
| -------------|:-------------:|:----------:|:----------:|
| Train        | 1115              | 1019           | 2134           |
| Validation   | 204              | 173           |  377          |
| Test         | ---              | ---           | 196169           |
Dataset statistics table

Below is another visual representation of the non-fish egg and fish egg classes

INSERT PICTURE HERE

##### Labeling
The labeling process for creating our datasets was conducted by PhD Biologist Candidate, Brian Stock, through a GUI interface coded in MATLAB. The data was randomly shuffled and Brian would
add a class category based on clusters of similar specimens seen in the dataset, resulting in our 18 classes. 

##### Preprocessing



### Part 4: Model
Based on this dataset, we decide to start off a standard CNN AlexNet. It consists of this many layers.

### Part 5: Object Size Detection
##### Computer Vision Algorithm

### Part 6: Conclusion
We now have a model and an object size detection to identify Nassau grouper.

Accuracy for fish egg/non-fish egg classes on validation set using AlexNet

- Accuracy: 97.61%
- Area Under Curve (AUC): 0.98

| Confusion Matrix        | ROC Curve           |
| ------------- |:-------------:|

Correctly predicted fish egg counts and size detection from test set after quality control from PhD biologist Brian Stock from the Semmens Lab at Scripps Institution of Oceanography
- / 3382 (%) predicted fish egg images
- / 3382 (%) valid object size detection
