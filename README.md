
# Intel Image Classification with Pytorch

![PyTorch Logo](https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png)

This project aims to classify images into six different scenes: buildings, forest, glacier, mountain, sea, and street. The images are categorized using deep learning techniques implemented in PyTorch. The dataset used for this project was directly downloaded from [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification) and was originally published for the Analytics Vidhya Intel Image Classification Challenge. We extend our gratitude to both Kaggle and Analytics Vidhya for providing this dataset.

## Table of Contents

- [Intel Image Classification with Pytorch](#intel-image-classification-with-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
      - [FiftyOne: Dataset Exploration and Visualization](#fiftyone-dataset-exploration-and-visualization)
          - [key Features of Fifty-One:](#key-features-of-fifty-one)
  - [Training Process](#training-process)
      - [Models used :](#models-used-)

## Overview

In this image classification project, we explored the performance of four different models for classifying the dataset into six distinct scenes: buildings, forest, glacier, mountain, sea, and street. Each model has distinct characteristics that we examined to understand their impact on classification accuracy and training convergence.

- Simple Model: The first model we used is a straightforward architecture for image classification. It consists of a `stack of convolutional layers followed by fully connected layers` for prediction.

- Sigmoid Activation Model: In the second model, we used the same architecture as the simple model but applied the `sigmoid activation function` throughout the network. This activation function introduces non-linearity, which can influence the model's decision boundaries.

- Tanh Activation Model: The third model shares the same architecture as the previous models but employs the `hyperbolic tangent (tanh) activation function` instead. Tanh is another non-linear activation function that can provide different characteristics to the model.

- Normalized and Dropout Model: For the fourth model, we incorporated two additional techniques: `batch normalization` and `dropout`. Batch normalization normalizes the activations in each layer, which can help stabilize and accelerate the training process. Dropout randomly deactivates certain neurons during training, preventing overfitting and improving generalization

## Dataset

#### FiftyOne: Dataset Exploration and Visualization

Fifty-One is an open-source library that empowers data scientists and machine learning practitioners with powerful tools for analyzing and visualizing machine learning datasets. While particularly well-suited for image and video data, Fifty-One can handle a variety of data types. Its main objective is to bridge the gap in the machine learning pipeline by providing comprehensive visual explorations, dataset comparisons, and label inspections, enabling users to gain a deeper understanding of their datasets.



###### key Features of Fifty-One:

1. Visual Exploration: Fifty-One offers robust visualization tools, particularly beneficial for image and video data. Interactive visualizations help uncover valuable insights.

2. Label Inspection: With Fifty-One, inspecting and verifying dataset labels becomes effortless, aiding in identifying potential labeling errors or inconsistencies.

3. Dataset Comparison: Users can easily compare different datasets or versions of the same dataset, facilitating analysis of data cleaning and augmentation effects.

4. Integration with Machine Learning Libraries: Fifty-One seamlessly integrates with popular machine learning libraries such as PyTorch and TensorFlow, simplifying its incorporation into existing workflows.

5. Scalability: Designed to handle large datasets, Fifty-One proves to be a practical tool for real-world machine learning projects.


    

https://github.com/erfanakk/Intel_classification/assets/87381197/1f9ca9ba-17b0-47ca-8b07-59da7a96f5fc



In this video, we showcase the capabilities of FiftyOne in enhancing the dataset curation process, analyzing class distributions.




## Training Process

#### Models used :
    1. simple model 
    2. simple model with norm and dropout
    3. modal sigmoid
    4. model tanh

[in progers]
