# MetricLearning

Metric Learning (similarity based on a distance metric) has been around for quite some time with several deep neural networks being predominant in the field. We evalauted these prodominant deep neural networks: Variational Auto-Encoders (VAEs), siamese networks, triplet networks, and variational auto-encoders combined with siamese or triplet networks. The networks utilized in the Jupyter notebooks are set up for GPU usage. 

## Convolutional Baseline
The convolutional baseline takes two inputs that does not make use of image encoding. 

## Siamese Network
The Siamese (SiameseNetwork.ipynb) network takes two inputs that is run through the same encoder, creating two low-dimensional representations. This network learns an absolute distance.

## Triplet Network
The triplet (TripletNetwork.ipynb) network is an extension of the siamese network where it takes three inputs that is run through the same encoder, creating three low-dimensional representations. This network learns a relative distance. 

## VAE Network
The VAE network(VAE.ipynb) is a type of neural network that takes only one input and does not learn any type of distance metric. Instead, the network is a generative model where it learns to create a low-dimensional representation in a latent space.

## VAE Siamese Network
The VAE Siamese (VAESiameseNetwork.ipynb) network is a hybrid of both the VAE and Siamese networks. This network takes two low-dimensional representations taken from the same VAE network and learns an absolute distance. 

## VAE Triplet Network
The VAE triplet (VAETripletNetwork.ipynb) network is a hybrid of both the VAE and triplet networks. This network takes three-low dimensional representations taken from the same VAE network and learns an relative distance.

## Data
Data (data.py) is preprocessed and split between three different sets: training, validation, and testing.

## Loaders
Loaders (loaders.py) are the networks defined by their structure e.g. triplet network taking three inputs, siamese taking two. 

## Networks
Networks (nets.py) formulate the general architecture of the networks such as the layers they use. 
