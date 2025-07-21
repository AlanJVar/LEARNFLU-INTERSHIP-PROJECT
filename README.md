# LEARNFLU-INTERSHIP-PROJECT

This project focuses on predicting customer churn using a neural network model trained on an Iranian customer dataset. It covers the complete machine learning pipeline, including Exploratory Data Analysis (EDA), model training, and model evaluation.

Project Overview
Customer churn is a critical issue for businesses, as retaining existing customers is often more cost-effective than acquiring new ones. This project aims to build a predictive model to identify customers who are likely to churn, allowing businesses to implement targeted retention strategies. A neural network is employed for this binary classification task, leveraging insights gained from comprehensive exploratory data analysis.

Dataset
The project utilizes a dataset named "Customer Churn.csv". This CSV file is expected to contain various customer attributes and a 'Churn' column indicating whether a customer has churned (1) or not (0).

Please ensure the Customer Churn.csv file is placed in the root directory of the project.

Key Steps:

Data Loading: Reads Customer Churn.csv into a Pandas DataFrame.

Initial Inspection: Prints df.head(), df.info(), and df.describe() to understand the data structure, types, and basic statistics.

Missing Values Visualization: Generates a heatmap to visualize any missing values in the dataset. (Note: The provided code comments suggest there are no null values, leading to a singular color screen).

Churn Distribution: Creates a count plot to show the distribution of churned vs. non-churned customers.

Feature Distributions: Generates histograms for all numerical features to visualize their distributions.

Pairplot: Creates a pairplot, colored by 'Churn', to visualize relationships between pairs of features and their distributions.

Correlation Heatmap: Displays a heatmap of the correlation matrix between features, helping to identify highly correlated variables.

Categorical Encoding: Converts categorical variables into numerical format using one-hot encoding (pd.get_dummies).

Feature Scaling: Applies StandardScaler to normalize numerical features, which is crucial for neural network performance.

Data Splitting and Saving: The script implicitly prepares X (features) and y (target) and is expected to save X_train.npy, X_test.npy, y_train.npy, y_test.npy for subsequent model training. (Note: The provided EDA1001.py snippet shows the creation of X and y but not the saving to .npy files. This step is inferred from ModelTraining1001.py loading these files).

To run EDA:

Bash

python EDA1001.py
Model Training
The ModelTraining1001.py script builds, compiles, and trains a Sequential Neural Network model.

Key Steps:

Load Preprocessed Data: Loads X_train.npy, X_test.npy, y_train.npy, and y_test.npy generated from the EDA step.

Neural Network Architecture:

A Sequential model with three Dense layers.

The first Dense layer has 64 units with relu activation, taking the input shape of the training data.

Two Dropout layers (rate 0.3) are included after the first two Dense layers to prevent overfitting.

A second Dense layer has 32 units with relu activation.

The final Dense layer has 1 unit with sigmoid activation, suitable for binary classification.

Model Compilation:

Optimizer: adam

Loss Function: binary_crossentropy (standard for binary classification)

Metrics: accuracy

Model Training: Trains the model using model.fit() with:

epochs=100

batch_size=32

validation_data=(X_test, y_test) for monitoring performance on unseen data during training.

Model Saving: Saves the trained model as churn_nn_model.h5.

Training History Plots: Generates and displays plots for:

Training and Validation Accuracy over epochs.

Training and Validation Loss over epochs.

To train the model:

Bash

python ModelTraining1001.py
Model Evaluation
The ModelEvaluation1001.py script evaluates the performance of the trained neural network model.

Key Steps:

Load Data and Model: Loads X_test.npy, y_test.npy, and the saved churn_nn_model.h5.

Prediction: Makes predictions on the test set (X_test) and converts probabilities to binary class labels (0 or 1) using a threshold of 0.5.

Accuracy Score: Calculates and prints the overall accuracy of the model.

Classification Report: Prints a detailed classification report, including precision, recall, f1-score, and support for each class.

Confusion Matrix: Computes and plots a confusion matrix as a heatmap, providing a clear visualization of true positives, true negatives, false positives, and false negatives.

To evaluate the model:

Bash

python ModelEvaluation1001.py
Results
After running the respective scripts, you will observe:

EDA: Various plots (heatmap, countplot, histograms, pairplot, correlation heatmap) will be displayed, providing insights into the dataset.

Model Training: Plots showing the training and validation accuracy and loss will be displayed, indicating the model's learning progress. The churn_nn_model.h5 file will be created.

Model Evaluation: The console will output the model's accuracy and a classification report. A confusion matrix heatmap will be displayed, visually summarizing the model's performance on the test set.

Future Enhancements
Hyperparameter Tuning: Experiment with different neural network architectures, number of layers, units per layer, dropout rates, learning rates, and optimizers to optimize model performance.

Cross-Validation: Implement k-fold cross-validation for more robust model evaluation and to reduce bias from a single train-test split.

Feature Engineering: Create new features from existing ones that might better capture customer behavior related to churn.

Advanced Preprocessing: Explore different scaling techniques or handle outliers.

Imbalanced Data Handling: If the churn class is highly imbalanced, consider techniques like SMOTE, oversampling, or undersampling to improve minority class prediction.

Contact
For any questions or inquiries, please contact me at alanvarghese852@gmail.com.
