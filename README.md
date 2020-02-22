#### dsc-4-final-project-online-ds-pt-112618

# Module 4 Final Project - Predicting Breast Cancer using Machine and Deep Learning


## Overview

Within a given dataset, can breast cancer be accurately predicted?

To acheive this Deep Learning (in addtion to Machine Learning) will be used for this project.  Deep learning is quite complex and is a subset of machine learning. Quite simply, it’s a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. In deep learning, a computer model learns to perform classification or regression tasks directly from images, texts, or sounds.  Moreoever, these models can achieve state-of-the-art accuracy.

## Problem Statement

Analyze dataset to identify which patients have breast cancer and use Machine Learning and Deep Learning models to determine which model best predicts breast cancer.

## Data Source Overview

Breast cancer data obtained from University of California, Irvine.  The data contains 569 patient samples and was stained to determine if the patient had cancer (i.e. - malignant) or no cancer (i.e. – benign)

The dataset was obtained from Kaggle: : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

## Scrubbing the Data

Again, there were 569 rows of data, which corresponded to each individual patient. Additionally, there were 33 columns or features to the data (such as ID, diagnosis, and details about the FNA masses). 

There was a completely useless column called Unnamed 32, which had nothing but null values. This was dropped from the dataset, so it wouldn’t skew the analyses. I also noticed that the ID column was comprised of integer values. Since this column was just full of patient ID numbers, it wouldn’t be included in my analyses either.

## Exploratory Data Analysis

Analyzing the histograms of the masses showed the distribution, but nothing really stood out. I checked a few scatter graphs looked at some of the mass features (area mean and symmetry worst) and I noticed many of the masses were clustered with a mean of 500 and symmetries between 0.2 to 0.3. Since I am interested in the number of patients with either malignant or benign tumors, I made a bar graph to visualize this. I saw out of the 569 patients, 212 had breast cancer malignancies and 357 had benign tumors.

Next step was to transform the malignant and benign columns, which were categorical values. Categorical values can not be analyzed properly in the machine and deep learning models, so changing them to 1 and 0 (i.e. – malignant or benign) was the next step. I did run another scatter plot of all the columns against the diagnosis column to see if anything stood out. This generated a massage scatter plot of all the features, so I scaled it down. What stood out mainly for me visually was the malignant tumors were much larger in terms of perimeter, area, radius than the benign ones. Also, the malignant tumors generally had more texture to them. Then I generated a correlation table and heat map to look at all the features. I ran it out of curiosity to see if any other features were worth examining. I didn’t see anything else and furthermore, for this project, I was interested in predictive modeling looking at the breast cancer diagnosis.


<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/hr_histogram_plots.png'>

<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/mod%204%20scatter%20plot.png'>

<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/mod%204%20tumor%20types.png'>

<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/mod%204%20tumor%20scatter%20plot.png'>



## Predictive Modeling

Again, I was solely interested in which models would best predict breast cancer malignancy. My independent data will tell me the features that can detect if a patient has cancer, and my dependent data will tell me if a patient has cancer or not. I decided to use an 80% training set and 20% testing set. Next using the StandardScaler feature in Sklearn, I scaled the data. Scaling the data brings all features to the same level of magnitude. So, the data will be within a specific range for example 0 -100 or 0 – 1.

### Machine Learning Models

Logistic Regression performed the best out of all the Machine Learning models. A close second was Random Forest at 96%. The model accuracy was the following:

* 	K-Nearest Neighbors (kNN): 94%
* 	Logistic Regression: 97%
* 	SVM: 95%
* 	Decision Tree: 94%
* 	Random Forest: 96%

<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/CM_log%20reg.PNG'>

<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/CM_DT.PNG'>

### Deep Learning

I decided to use Keras for modeling my data. Keras contains numerous implementations of commonly used neural network building blocks such as layers, objectives, activation functions, optimizations, and a host of tools to make working with image and text data easier. What I like about Keras is it’s great for beginners, minimalistic, and its modular approach makes it easy to get deep neural networks up and running.

After compiling the metrics and layers for the neural network, I ran the Keras model. The final output showed the model had 95% accuracy in predicting breast cancer malignancies.

<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/CM_Keras.PNG'>

## Conclusions

* Logistic regression had the highest level of accuracy (97%) in predicting malignancy 

* Random Forest and Keras came in second (96%) in accurately predicting malignancy

* This methodology could help reduce physician burnout and speed up detection

* Accurate detection can reduce the likelihood of a missed diagnosis by a human eye

* Conversely, this methodology isn’t perfect. A physician will still have to double check positive results

## Future Work and Recommendations

* Use Logistic regression for detection, and improve parameters in Keras deep learning model to improve accuracy

* Possibly adjust training and testing set of data to see if that will yield higher accuracy

* Add more features to the dataset, making it more robust for testing

* Explore other deep learning models to determine level of accuracy

