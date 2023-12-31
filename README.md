# Credit Card Fraud Detection

## Overview
This repository contains Python code for a Credit Card Fraud Detection project using a Random Forest classifier. The project aims to identify fraudulent transactions within a credit card dataset through machine learning techniques.

## Dataset
The dataset used in this project is sourced from a CSV file (`creditcard.csv`). It contains features related to credit card transactions, including a "Class" column indicating whether a transaction is fraudulent (Class=1) or not (Class=0).

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Install the required packages using:

 - pip install numpy pandas matplotlib seaborn scikit-learn
## Code Structure
 - credit_card_fraud_detection.py: Main Python script containing the code for loading the dataset, data exploration, building a Random Forest classifier, and evaluating the model's performance.  
 - README.md: This file, providing an overview of the project, dataset, dependencies, instructions for running the code, and additional details.
## Usage
 1. Clone the repository:  
  git clone https://github.com/yourusername/credit-card-fraud-detection.git  
  cd credit-card-fraud-detection  
 2. Install dependencies:  
  pip install -r requirements.txt  
 3. Run the script:  
  python credit_card_fraud_detection.py
## Results
The script generates visualizations, prints dataset statistics, and evaluates the Random Forest classifier's performance using metrics such as accuracy, precision, recall, F1-score, and the Matthews correlation coefficient. A confusion matrix is also displayed.

## Additional Details
 - The **data** directory contains the dataset file (**"creditcard.csv"**).  
 - The **images** directory stores visualizations generated by the script.  
 - Adjust the **test_size** and **random_state** parameters in the script for different train-test 
 - splits and reproducibility.  
 - Explore the Jupyter Notebook version (**"credit_card_fraud_detection.ipynb"**) for an 
   interactive experience.
## References
 - Scikit-learn Documentation
 - Pandas Documentation
 - Matplotlib Documentation
 - Seaborn Documentation
