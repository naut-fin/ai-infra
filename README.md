# ai-infra
Credit Default Prediction using Neural Networks
Overview

This project implements an end-to-end binary classification pipeline to predict credit card default using a neural network built with TensorFlow/Keras.

The goal of this exercise was to understand the complete machine learning workflow, including:

data preprocessing

train / cross-validation / test splits

neural network architecture selection

hyperparameter tuning

regularization experiments

early stopping to prevent overfitting

threshold tuning

evaluation using precision, recall, F1 score, and ROC-AUC

This project focuses on understanding model training behavior and evaluation, rather than maximizing leaderboard performance.

Dataset

Dataset used:

UCI Default of Credit Card Clients Dataset

This dataset contains demographic, credit history, and payment information for credit card holders, with the goal of predicting whether a client will default on the next payment.

Number of samples: 30,000

Target variable:

default payment next month

Binary classification:

0 → No default
1 → Default
Project Structure
credit-default-neural-network/
│
├── notebooks/
│   └── credit_default_nn.ipynb
│
├── requirements.txt
│
└── README.md
Machine Learning Workflow
1. Data Preprocessing

Steps performed:

dataset loading

feature / label separation

train / validation / test split

feature scaling using StandardScaler

Data split:

Training set
Cross Validation set
Test set

This allows:

training → learn model parameters
cross validation → tune hyperparameters
test → final evaluation
Neural Network Architecture

The best performing architecture after experimentation was:

Input Layer
Dense(32, ReLU)
Dense(16, ReLU)
Dense(1, Linear)

Loss function:

BinaryCrossentropy(from_logits=True)

Optimizer:

Adam
learning_rate = 0.001

Early stopping was used to prevent overfitting.

Hyperparameter Tuning
Architecture Search

Multiple architectures were tested:

32 → 16
64 → 32
128 → 64
256 → 128
512 → 256

Selection criteria:

lowest cross validation error

Best architecture:

Dense(32) → Dense(16)
Regularization Experiments

L2 regularization values tested:

λ = [0, 0.01, 0.1, 1]

Observation:

λ = 0 performed best

This indicates the model was not significantly overfitting, so additional regularization was unnecessary.

Threshold Optimization

Default classification threshold:

0.5

Because the dataset is imbalanced, different thresholds were evaluated:

[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

Metrics evaluated:

precision
recall
F1 score

Best threshold (for F1):

0.3
Model Evaluation

Final test metrics:

Accuracy ≈ 78%
AUC ≈ 0.728

Interpretation of AUC:

There is a 72.8% probability that the model ranks a random defaulter
higher than a random non-defaulter.

This performance is consistent with baseline models for this dataset.

Experiments Conducted

The following experiments were performed:

neural network architecture search

regularization tuning

early stopping

class imbalance handling using class weights

threshold tuning

ROC-AUC evaluation

Observation:

Using class weights slightly decreased AUC in this dataset.

Key Learnings
1. Validation loss is critical

Monitoring validation loss helps detect overfitting during training.

2. Accuracy alone is insufficient

Because the dataset is imbalanced, metrics like:

precision
recall
F1 score
AUC

provide more meaningful insights.

3. Model ranking vs classification threshold

A model produces probabilities, not decisions.

The threshold determines the decision policy, which can vary depending on business needs.

4. Regularization is not always beneficial

Regularization is helpful when overfitting exists, but unnecessary regularization can reduce model performance.

Future Improvements

Possible improvements include:

experimenting with deeper neural networks

comparing performance with tree-based models (Random Forest / XGBoost)

using feature engineering techniques

hyperparameter optimization using automated search methods

Purpose of This Project

This project was built as part of learning the fundamentals of neural network training and evaluation, with a focus on understanding the complete ML workflow.

The long-term goal is to apply this knowledge toward building and operating AI/ML infrastructure systems.

Technologies Used
Python
TensorFlow / Keras
NumPy
Pandas
Scikit-learn
Matplotlib
Jupyter Notebook

License
This project is for educational purposes.
