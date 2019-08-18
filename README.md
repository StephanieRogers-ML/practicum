#  Project Background  
##  Image-based & Object-based Deep Learning for Vehicle Interiors 
[Open in Colab](https://colab.research.google.com/github/StephanieRogers-ML/practicum/blob/master/BinaryClassification.ipynb) 
Binary Classification with EfficientNet

[Open in Colab](https://colab.research.google.com/github/StephanieRogers-ML/practicum/blob/master/Classification.ipynb) 
Multi-class Classification with MobileNet

[Open in Colab](https://colab.research.google.com/github/StephanieRogers-ML/deep-learning_vehicle-dashboard/blob/master/FinalDetection.ipynb) 
SSD Detection with MobileNet Feature Extractor


Modern Dashboards can accommodate a broad array of gauges, climate controls, infotainment, and entertainment systems, which contrast to the earlier simpler controls of only speed, fuel, and oil pressure. Automakers have different and distinct ways of designing their dashboards and using a collection of visual identification tools, machine learning and deep learning algorithms we will learn to leverage vehicle features of the dashboard to identify the vehicle and its characteristics. 

The goal of this project is to develop a machine learning model that can help identify the content of a vehicle and its features using visual cues from its interior images of the vehicle. This is a multiclass supervised classification problem that will require labeled images to learn the features from curves, edges, and combination of features.  Our dataset consist of images collected from the CompCar dataset.
![Future](https://github.com/StephanieRogers-ML/deep-learning_vehicle-dashboard/blob/master/Sample_Data/futuristic_car.jpg)

##  Deep Learning Models

### Objectives
* [ ]  Vehicle Classification: Previously trained on ImageNet.
* [ ]  Object detection: model that localize and identify multiple objects in a single image.
* [ ]  Demo
 

### Vehicle_Classification

This repo was created as a summer practicum project.  Image Classification is applied to interior vehicle images from 3 different Makes using [tensorflow.js](https://js.tensorflow.org) and Google Colab. It is loosely based on the [tfjs Mobilenet example](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet).

### Object_Detection 
## TensorFlow.js Demo Example

1.  Image Classification done using a pre-trained model as the base and different classifiers, feature extractors, and fine tuning on a custom dataset. 
        classifier_url = 
        feature_extractor_url =
   
2.  Object Detection
    Faster RCNN Inception V2 coco   
    Single-Shot Multibox Detector (SSD) with feature extraction head from MobileNet   
    SSD Lite Mobilenet V2
    SSD Mobilenet V2


 
## Summary of Benchmarks  


|    Model      | Interior  |  steering |  odometer |   control |   gear    | 
|:-------------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|EfficientNet   |   0.946   |   0.885   |   0.804   |   0.906   |   0.857   |   
|MobileNet      |   0.953   |   0.949   |   0.259   |   0.777   |   0.789   |  
|SSD & MobileNet|   0.710   |   0.521   |   0.507   |   0.680   |   0.656   |   




##  Product Demo    


### Vehicle Detection & Classification
* [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) 
* [Tensorflow COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd/demo) 
* [Steering Ecosystem Simulation Demo](https://shiffman.github.io/Tensorflow-JS-Examples/05_neuro_evolution_steering/) 


##  Contact us  

Stephanie Rogers | gp5880@wayne.edu  | stephanierogers.ml@gmail.com  

Dante Burch      |                    |

Jatin Gongiwala  | jatinmg97@wayne.edu |  Jatinmg97@gmail.com  

Ranjitha Vidyashankar  | gm8135@wayne.edu | ranjithavshankar@gmail.com  
