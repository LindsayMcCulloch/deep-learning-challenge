
## Overview 

The purpose of this analysis is to build a deep neural network model that can predict whether crowdfunding projects will be successful. By preprocessing the data, creating a neural network model, training it on the training data, and evaluating its performance on the test data, the analysis aims to develop a predictive model that can assist in identifying successful projects in advance. This analysis can help stakeholders make informed decisions and allocate resources effectively to maximize the success rate of projects on the crowdfunding platform.

## Results

# Data Preprocessing

**What variable(s) are the target(s) for your model?**

The target variable for the model is the "IS_SUCCESSFUL" column in the dataset

**What variable(s) are the features for your model?**

The features for the model include all the columns in the dataset except for the target variable "IS_SUCCESSFUL." 

**What variable(s) should be removed from the input data because they are neither targets nor features?**

In the preprocessing steps, the "EIN" column was dropped from the input data as it is an identification column and not a relevant feature for the model. Additionally, the "NAME" column was also dropped as it does not provide meaningful information for predicting the success of projects on the crowdfunding platform.

# Compiling, Training, and Evaluating the Model

**How many neurons, layers, and activation functions did you select for your neural network model, and why?**


For the neural network model in this analysis, an input layer was used with 19607 neurons based on the number of input features, 3 hidden layers consisting of; 100 neurons (ReLu), 30 Nuerons (tanh) and 10 neurons (sigmoid), and an output layer of 1 nuerons (sigmoid) allowing the model to learn the patterns of the data Activation Functions:

ReLU (Rectified Linear Unit) was chosen for the first hidden layer as it is a common choice for hidden layers due to its effectiveness and simplicity.
tanh (Hyperbolic Tangent) activation function was selected for the second hidden layer to introduce non-linearity and capture complex patterns in the data.
Sigmoid activation function was used for the third hidden layer to introduce non-linearity and map the output to a probability between 0 and 1.
Sigmoid activation function in the output layer is suitable for binary classification tasks like predicting project success.
These choices were made to introduce non-linearity, capture complex patterns, and optimize the model's performance for the binary classification task of predicting project success on the crowdfunding platform.

**Were you able to achieve the target model performance?**

The model's loss is 0.4805, and the accuracy is 0.7847. This means that the model achieved an accuracy of approximately 78.47% on the test dataset.

Based on the target set to achieve higher than 75% accuracy, the model has successfully met that goal with an accuracy of 78.47%. Therefore, the model has achieved the desired target performance.


**What steps did you take in your attempts to increase model performance?**

To increase the model performance and achieve an accuracy higher than 75%, several steps could have been taken. Some common strategies to improve model performance include:

Hyperparameter Tuning: Adjusting hyperparameters such as learning rate, batch size, number of epochs, optimizer choice, activation functions, and model architecture can significantly impact model performance.

Regularization: Implementing techniques like L1 or L2 regularization, dropout, or batch normalization can help prevent overfitting and improve generalization.

Feature Engineering: Creating new features, scaling or normalizing data, handling missing values, and encoding categorical variables can enhance the model's ability to learn complex patterns.

Model Architecture: Experimenting with different neural network architectures, including the number of layers, types of layers, and units in each layer, can lead to better performance.


## Summary


The deep learning model achieved an accuracy of 78.47% on the test dataset, surpassing the target of 75% accuracy. This indicates that the model performed reasonably well in classifying the data.

For this classification problem, considering the success of the deep learning model, it may be beneficial to explore other advanced machine learning models such as Gradient Boosting Machines (GBM) or Random Forest.

Recommendation:

Gradient Boosting Machines (GBM):

GBM is an ensemble learning method that builds multiple decision trees sequentially, where each tree corrects the errors of the previous one.
GBM is known for its high accuracy, robustness to overfitting, and ability to handle complex relationships in data.
It can be effective in handling tabular data and is widely used in various machine learning competitions and real-world applications.
Random Forest:

Random Forest is another ensemble learning method that builds multiple decision trees and combines their predictions through voting or averaging.
It is known for its simplicity, scalability, and ability to handle high-dimensional data with ease.
Random Forest is robust to overfitting and noise in the data, making it a good candidate for classification tasks.
Explanation:

While the deep learning model performed well in achieving the target accuracy, exploring models like GBM or Random Forest can provide additional insights and potentially improve performance further. These models offer different strengths and weaknesses compared to neural networks, and their ensemble nature can capture complex patterns in the data effectively. By experimenting with these models and comparing their performance, you can determine the most suitable approach for this classification problem.