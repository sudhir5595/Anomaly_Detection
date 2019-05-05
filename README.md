# Anomaly_Detection
In this project we are going to detect anomalies. Footage or data that we are using will be of stationary camera mounted at pedestrian walkways.The crowd density in the walkways was variable, ranging from sparse to very crowded. Abnormal crowd event can be classified into two groups on the basis of a scale of interest: global anomalous event and local anomalous event. The global anomalous event (GAE) concentrates on deciding if the whole scene is abnormal or not, whereas the local anomalous event (LAE) determines the regions where the abnormal event is happening within the scene.

In this project we are trying to detect LAE from the video.

## Evaluation
The evaluation is done by F1 Score.
<img src = "images/f1-score.jpg" width = "300" >

## Approaches 
We followed two different approaches to detect the LAE from the scene and finally applied deep learning to detect anomaly. 

## Dataset
We have used [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz) in this project. It consists of dataset which was split into 2 subsets, each corresponding to a different scene. The video footage recorded from each scene was split into various clips of around 200 frames.

Peds1: clips of groups of people walking towards and away from the camera, and some amount of perspective distortion. Contains 34 training video samples and 36 testing video samples. 

Peds2: scenes with pedestrian movement parallel to the camera plane. Contains 16 training video samples and 12 testing video samples.


