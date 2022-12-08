# CSCI-7850-Stellar-Classification

Codes for Stellar Classification final project that incorporates data ablation and standardization based on the Sloan Digital Sky Survey data release compiled and found at Kaggle.com under Stellar Classification Dataset - SDSS17 https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

Three codes that use deep learning neural networks have been included here, a flat, wide, and deep network.

Data ablation and standardization are performed in each neural network by setting the "ablate" and "standardize" variables near the top to True.

The data file used in this project is located at the following address or above at the Kaggle.com address: https://csci-7850-stellar-classification.nyc3.digitaloceanspaces.com/star_classification.csv

Recommend using the wget command to download the data file to use in the codes like so:
wget https://csci-7850-stellar-classification.nyc3.digitaloceanspaces.com/star_classification.csv

# Demonstration notebook file

A IPython notebook file has been included to illustrate the results of one of the neural networks. In this case, the Wide neural network with data ablation and standardization. 

Have the data file located in the same directory as the notebook file to execute and interact with the notebook.
