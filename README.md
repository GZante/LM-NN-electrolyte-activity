# Activity Coefficients Prediction using a Levenberg-Marquardt Neural Network

This repository contains a Python script for predicting activity coefficients using a Levenberg-Marquardt Neural Network (LM-NN). The script uses the Pyrenn library for neural network creation and training, and Scikit-learn for data preprocessing and evaluation metrics.


## Introduction

This script is designed to predict activity coefficients of electrolytes in binary solutions using a neural network. The neural network is trained on a dataset containing activity values and various descriptors. The script iterates through different neural network configurations to find the best one based on the R-squared metric.

## Requirements

To run this script, you need the following Python libraries:

- `pyrenn`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Data

The script assumes that the Excel file contains activity values (activity of water or activity of the electrolyte or osmotic coefficient for multiple salts). The column "Activity" contains activity values, and other columns include molal concentration of electrolyte and descriptors (e.g., Pitzer parameters). Pitzer parameters were found to be the descriptors that led to the most accurate results. 

For more details, refer to the paper:

"Machine learning for determination of activity of water and activity coefficients of electrolytes in binary solutions", Artificial Intelligence Chemistry, Volume 2, Issue 1, June 2024, 100069, DOI: 10.1016/j.aichem.2024.100069

## Results

The script prints the best neural network configuration along with the evaluation metrics:

Hidden Neurons: Number of neurons in the hidden layer
R2: R-squared value
MAE: Mean Absolute Error
RMSE: Root Mean Squared Error
AARD: Average Absolute Relative Deviation

Best results that obtained by selecting multiple electrolytes (a few hundred datapoints) of electrolytes of the same family (sulfates, chlorides, nitrates...).  
