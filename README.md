# Metric Learning

Metric Learning aims to learn a metric that allows for measuring similarity and/or dissimilarity between instances. This repository contains jupyter notebooks for the following 6 types of networks: a seperable convolutional classifier, variational auto-encoders, Siamese networks, triplet networks, and variational auto-encoders combined with Siamese or triplet networks.

These networks are evaluated on the EMNIST balanced dataset.

# Requirements
Tensorflow 2.0
tensorflow_datasets (python package)

To install Tensorflow 2.0, follow the instructions in the following link: https://www.tensorflow.org/install
To install tensorflow_datasets, use the following command:
'pip install tensorflow-datasets'

# Files
Each jupyter notebook has the name of the network utilized within.
Our jupyter notebooks are set up for GPU usage.
For simplicty, we list them here:

ConvolutionalBaseline.ipynb - our seperable convolutional classifier; does not utilize metric learning
VAE.ipynb - our variational auto-encoder network.
SiameseNetwork.ipynb - our Siamese network.
TripletNetwork.ipynb - our triplet network (based on the siamese architecture.
VAESiameseNetwork.ipynb - our hybrid of the variational auto-encoder and siamese network.
VAETripletNetwork.ipynb - our hybrid of the variational auto-encoder and triplet network.

These jupyter notebooks train and evaluate these networks on the EMNIST balanced dataset.

Our .py files contain the building blocks of our networks.
We list their purposes here:

data.py - loads our EMNIST balanced dataset 
loaders.py - generates batches of pairs/triplets of instances from the EMNIST balanced dataset
nets.py - contains the architectures of the networks, the loss functions, and the evaluation code.

# License

Academic use: The software is provided as-is under the GNU GPLv3.
Any restrictions to use for-profit or non-academics: License needed.
