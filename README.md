# Project 1 CSEC 620 
## Detecting Intrusions/Attacks On A Network Via A Neural Network Examining Packets
By Nikhil Patil

## Resources Used
1. https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#numerical_columns
- Provided an excellent basis on how to begin this project as well as the basis for my first model
2. https://docs.dataprep.ai/user_guide/clean/clean_df.html
- This was a Python library that I found about that is incredibly useful for cleaning and standardizing a Pandas Dataframe. It also decreases the size of the dataframe in memory. <br/>
3. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- This was also useful as well to encode the label column rather than manually encoding them.
- Additionally, not listed here is the fit_transformer. This was excellent to normalize the data and make it easier for the neural network to work with.
4. https://www.scitepress.org/papers/2018/66398/66398.pdf
- This was the paper that provided the dataset as well as the information needed to complete this project

## How to run 
I created this model locally on my PC which is running Windows 11. I used the Pycharm Professional IDE which is incredibly useful as for each project you can create a virtual python environment as well a virtual conda environment. I am using python version 3.10.3 as this was the version that had the least amount of conflicts with tensorflow and other conda packages I used to construct the models.<br/>
After installing the various dependencies that can be seen at the top of the jupyter notebooks, the program should run fine. It is also necessary to install CUDA as well, otherwise Tensorflow will not run on the GPU (that is if you have a NVIDA GPU). I do not have a NVIDIA GPU, so I uploaded the models to the granger GPU cluster in order to obtain my results. More detailed instructions can be seen in the <a href="https://docs.google.com/document/d/1LuK7qttWaMectmBl55IEa5gfQykr27vcx9sM3Von3h4/edit?usp=sharing">Project 1: Deep Neural Network for Malicious Traffic Analysis</a>. That Google Doc is discussion of results and comparison of my results as well. Albeit similar to the README, it is different in the content that it offers.  

## Goal
The goal for this model was to achieve a similar detection rate to those seen in the paper. At first, I was not able to do this as my model suffered greatly from performance issues whether it be execution time or accuracy. My original model had an accuracy of .65 and execution time of 6450 seconds. The model was not effective at all and this was training only on 4 features out of 77. This is why I chose to create a new model from scratch that preps and cleans the data prior to being processed and then use a larger model that scales down to 16 neurons/nodes(?). 

## First Model
My first model was terribly inefficient. It uses a functional model rather than a sequential model. It also normalizes the data and coverts a pandas dataframe to a dataset to be used by tensorflow. It uses only 4 features that were listed by the researchers as the best ones to be used to detect DDoS attacks according to the paper. 

## Random Forest
Random Forest was the "best" algorithm used by the researchers in the paper. I decided to improve my first model by implementing Random Forest. It was then brought to my attention that Random Forest is NOT a neural network model but rather a machine learning algorithm. The way I implemented into my program is via scikit-learn as this has a native Random Forest implementation. This severely increased the performance of my program, as well as the accuracy.  

## Second Model
The second model was made from scratch taking what I've learned from my previous endeavors with this project. I decided to use a sequential model and go with a larger model (more neurons and more layers). I also use dataprep to clean the dataset before it is used to reduce the size of it in memory. I believe that because I increased the size of my model a significant degree, the accuracy is far greater with around .99 detection rate of a DDoS attack. Of course, this is a binary dataset (i.e. packets can be either DDoS or Benign), this makes classification far easier. My reasoning for using a sequential model is that in the in-class example with the PIMAS dataset, it is used, and albeit on a smaller dataset, it appeared to more efficient than a functional model. For this and First Model, ADAM was the optimizer used, as it seems to be the best general optimizer. 