# Credit Card Fraud Prediction

## Overview
This repository contains Python code for a Credit Card Fraud Prediction project. The goal is to predict fraudulent credit card transactions using machine learning algorithms. The code includes data loading, exploratory data analysis, feature engineering, model training, and evaluation.

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data](#data)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Results](#results)

## Introduction
Credit card fraud is a significant concern for financial institutions. This project aims to develop a machine learning model to predict fraudulent transactions. Three different models are implemented and evaluated: Logistic Regression, Support Vector Machine (SVM), and Decision Tree.

## Dependencies
- pandas
- seaborn
- numpy
- matplotlib
- scikit-learn

Install the required dependencies using the following command: \n
pip install pandas seaborn numpy matplotlib scikit-learn

## Data
The dataset used for this project is loaded from the "creditcard.csv" file. It contains information about credit card transactions, including features like transaction amount, time, and various V1 to V28 features.

## Exploratory Data Analysis
The initial analysis includes checking the class distribution of the target variable, exploring correlations between features, and visualizing feature importance using a heatmap and bar chart.

## Feature Engineering
Feature engineering involves dropping duplicated values, splitting the dataset into input (X) and target (Y) variables, and applying Principal Component Analysis (PCA) to reduce the number of independent variables.

## Model Training
Three different models are trained: Logistic Regression, Support Vector Machine (SVM), and Decision Tree. Hyperparameter tuning is performed using RandomizedSearchCV to find the optimal parameters.

## Evaluation
Model performance is evaluated using metrics such as accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC.

## Results
The final results are presented in a DataFrame, summarizing the performance of each model based on different metrics.

