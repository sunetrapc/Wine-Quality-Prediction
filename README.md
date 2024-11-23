# Wine Quality Prediction

This project involves building a machine learning model to predict wine quality based on various chemical and physical features. The dataset contains attributes such as alcohol content, acidity, and sugar levels, along with the wine quality rating.

## Project Overview
The goal of this project is to predict the quality of wine using a deep learning model. The approach involves:
1. Data exploration and preprocessing.
2. Building a neural network model using Keras/TensorFlow.
3. Evaluating the model's performance using appropriate metrics.

## Dataset
The dataset used in this project is `wine quality with category value.xlsx`, which includes various features of wine and their corresponding quality ratings.

## How to Run the Model
### Prerequisites
To run this model, you will need the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
- openpyxl (for reading Excel files)

## Findings
Feature Correlations: During exploratory data analysis (EDA), we observed that certain features, such as alcohol content and volatile acidity, showed a moderate correlation with wine quality, suggesting that these factors are likely more important in predicting the quality of wine. Features like citric acid and residual sugar had weaker correlations, indicating that they might not contribute as significantly to the predictions.

Data Distribution: The dataset showed that wine quality ratings were somewhat skewed, with more wines rated between "5" and "6". This distribution may affect the model's ability to predict very low or very high quality wines accurately
