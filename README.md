
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
        - [simple model](#simple-model)
        - [Simple Model with Norm and Dropout](#simple-model-with-norm-and-dropout)
        - [Sigmoid and Tanh Activation Models](#sigmoid-and-tanh-activation-models)
          - [Sigmoid Activation Model](#sigmoid-activation-model)
          - [Tanh Activation Model](#tanh-activation-model)
        - [Transfer Learning](#transfer-learning)
          - [Transfer Learning with ResNet-50:](#transfer-learning-with-resnet-50)
          - [Transfer Learning with MobileNet:](#transfer-learning-with-mobilenet)
        - [The Power of Ensemble:](#the-power-of-ensemble)

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
In this section, we present a comparison of the training and test accuracy for each of the four models used in our image classification project:

#### Models used :
    1. simple model 
    2. simple model with norm and dropout
    3. modal sigmoid
    4. model tanh
    

##### simple model 

 - The simple model demonstrates reasonable performance during training, achieving high accuracy on the training data. However, it appears to suffer from overfitting, as the accuracy on the test data is comparatively lower.


``` Conv2D -> ReLU -> Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLUConv2D -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> FC -> ReLU -> FC ```


Training and Test Accuracy:
    

![simple_model](content/simple_relu_acc.png)

In the graph above, the blue line represents the training accuracy, while the red line represents the test accuracy. The increasing gap between the two lines indicates the model's tendency to overfit the training data, resulting in reduced generalization to unseen data.

To improve the model's performance and mitigate overfitting, we may consider techniques like dropout, batch normalization, or early stopping during training. Experimenting with different hyperparameters could also help achieve a better balance between training and test accuracy.







##### Simple Model with Norm and Dropout

 -  By incorporating batch normalization layers and dropout regularization, we effectively alleviate overfitting, resulting in a more balanced training and test accuracy. While the training accuracy might be slightly lower compared to the simple model, this trade-off is beneficial as we achieve better generalization to unseen data and prevent overfitting issues.


``` Conv2D -> ReLU -> BatchNorm -> Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> BatchNorm -> Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> Conv2D -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> Dropout -> FC -> ReLU -> FC```



Training and Test Accuracy:


![simple_model](content/simple_drop_norm_acc.png)


n the graph above, the blue line represents the training accuracy, while the red line represents the test accuracy. The smaller gap between the two lines clearly indicates that the model is not overfitting to the training data. Instead, it demonstrates better generalization capabilities, allowing it to perform well on unseen test data.



##### Sigmoid and Tanh Activation Models

The sigmoid and tanh activation models, while having distinct characteristics, did not perform as well in terms of accuracy compared to the other models in our image classification project.

###### Sigmoid Activation Model

The sigmoid activation model uses the same architecture as the simple model but replaces the activation function with sigmoid for all layers, including the output layer.

``` Conv2D -> Sigmoid -> Conv2D -> Sigmoid -> MaxPool -> Conv2D -> Sigmoid -> Conv2D -> Sigmoid -> MaxPool -> Conv2D -> Sigmoid -> Conv2D -> Sigmoid -> MaxPool -> Flatten -> FC -> Sigmoid -> FC -> Sigmoid -> FC ```

Training and Test Accuracy:

![simple_model](content/Sigmoid_acc.png)


###### Tanh Activation Model

Similarly, the tanh activation model retains the same architecture as the simple model but replaces the activation function with tanh for all layers.

```Conv2D -> Tanh -> Conv2D -> Tanh -> MaxPool -> Conv2D -> Tanh -> Conv2D -> Tanh -> MaxPool -> Conv2D -> Tanh -> Conv2D -> Tanh -> MaxPool-> Flatten -> FC -> Tanh -> FC -> Tanh -> FC```

Training and Test Accuracy:

![simple_model](content/tanh_acc.png)


##### Conclusion
After conducting an extensive comparison, we found that each model showed unique strengths and weaknesses. The simple model with normalization and dropout exhibited improved performance, effectively handling overfitting. The sigmoid activation model displayed competitive results, while the tanh activation model provided distinct characteristics.

Based on this analysis, we recommend the model that best suits your specific requirements for image classification tasks. We hope this model comparison proves valuable for your machine learning endeavors.


![simple_model](content/combined_plot.png)



##### Transfer Learning

Transfer learning is a powerful technique in the field of deep learning that allows us to leverage the knowledge gained from pre-trained models on large datasets and apply it to new, similar tasks. When we freeze the weights of a pre-trained model, it means we prevent them from being updated during the training process for our specific task. By freezing the weights, we retain the learned representations from the original dataset, and these representations act as a strong feature extractor for our new data.
Freezing weights is particularly useful when we have limited data for the new task or when the new task is similar to the original task on which the pre-trained model was trained. In such cases, freezing prevents overfitting and helps the model to generalize better on the new data.
The frozen layers of the pre-trained model are like a feature extraction backbone, providing high-level representations of the input data. On top of these frozen layers, we add new layers specific to our task and train only these new layers. This approach enables us to fine-tune the model for the new task while preserving the valuable knowledge obtained from the pre-trained model.





###### Transfer Learning with ResNet-50:
ResNet-50, a deep convolutional neural network, stands tall as a versatile transfer learning option. By leveraging its pre-trained weights, we infuse our image classification task with the knowledge of millions of images. Watch as ResNet-50 adapts to our dataset, capturing intricate details with remarkable precision. 


```bash
class TransferResnet50(nn.Module)
    def __init__(self, num_classes=6):
        super(TransferResnet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        
        for param in self.resnet50.parameters():
            param.requires_grad = False
                
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet50(x)
```


Training and Test Accuracy:

![resnet50_acc](content/resnet50_acc.png)



###### Transfer Learning with MobileNet:
Next up is the cutting-edge MobileNetV3, a lightweight and efficient architecture designed for mobile and embedded devices. Its compact design doesn't compromise accuracy, making it a perfect choice for real-time applications. Witness MobileNetV3's ingenuity in understanding our scenes and delivering outstanding results!

```bash
class TransferMobileNet(nn.Module)
    def __init__(self, num_classes=6):
        super(TransferMobileNet, self).__init__()
        self.MobileNet = models.mobilenet_v2(pretrained=True)
        
        for param in self.MobileNet.parameters():
            param.requires_grad = False
                
        self.MobileNet.classifier[-1] = nn.Linear(self.MobileNet.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.MobileNet(x)
```


Training and Test Accuracy:

![mobilenet_acc](content/mobilenet_accpng)



##### The Power of Ensemble:
Ensemble learning is a powerful technique in machine learning where multiple models are combined to make more accurate and robust predictions. Instead of relying on the output of a single model, ensemble methods leverage the collective knowledge of multiple models to improve overall performance.
The key idea behind ensemble learning is that different models may have different strengths and weaknesses, and by combining their predictions, we can mitigate the shortcomings of individual models and exploit their complementary strengths. This can lead to a more accurate and reliable final prediction.
Ensemble learning is like having a diverse team of experts working together to solve a problem. Each member brings their unique expertise, and by combining their insights, the team can arrive at more accurate and reliable solutions.
Now, we embark on an ingenious journey by combining the forces of ResNet-50 and MobileNetV2 through ensemble learning. This dynamic duo brings a whole new level of accuracy and robustness to our image classification project. By fusing their predictions, we create an ensemble model that excels in scene recognition, embracing the strengths of both architectures. 🌟


```bash
class MyEnsemble(nn.Module)


    def __init__(self, modelA, modelB, num_class=6):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        


        self.fc1 = nn.Linear(num_class *2  , num_class)


    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)

        out = torch.cat((out1, out2), dim=1)
 
        x = self.fc1(out)
        return x
```

Training and Test Accuracy:

![Ensemble_cat_acc](content/Ensemble_cat_acc.png)

