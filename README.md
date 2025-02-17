# deep-learning-challenge
This is the repository for Monash University Data Analytics Bootcamp Module 21 Challenge

## Contents

* `Analysis_Report.md` is the written analysis report for this challenge 
* `.png` files containing the screenshots of the notebook code used in `Analysis_Report.md`
* `AlphabetSoupCharity_Optimisation.ipynb` Jupyter notebook containing the Preprocessing machine learning model from the starter code provided 
* `AlphabetSoupCharity.h5`
* `AlphabetSoupCharity_Optimisation.ipynb` Jupyter notebook containing the optimised machine learning model for this challenge 
* `AlphabetSoupCharity_Optimization.h5`

# Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organisation, such as:

* **EIN and NAME:**Identification columns

* **APPLICATION_TYPE:** Alphabet Soup application type

* **AFFILIATION:** Affiliated sector of industry

* **CLASSIFICATION:** Government organisation classification

* **USE_CASE:** Use case for funding

* **ORGANIZATION:** Organisation type

* **STATUS:** Active status

* **INCOME_AMT:** Income classification

* **SPECIAL_CONSIDERATIONS:** Special considerations for application

* **ASK_AMT:** Funding amount requested

* **IS_SUCCESSFUL:** Was the money used effectively

# Instructions

## Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:

    * What variable(s) are the target(s) for your model?
    * What variable(s) are the feature(s) for your model?

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, `Other`, and then check if the replacement was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

7. Split the preprocessed data into a features array, `x`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

## Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organisation will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10 Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

## Step 3: Optimise the Model

Using your knowledge of TensorFlow, optimise your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimise your model:

* Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  
    * Dropping more or fewer columns.

    * Creating more bins for rare occurrences in columns.

    * Increasing or decreasing the number of values for each bin.

* Add more neurons to a hidden layer.

* Add more hidden layers.

* Use different activation functions for the hidden layers.

* Add or reduce the number of epochs to the training regimen.

***Note:*** *If you make at least three attempts at optimising your model, you will not lose points if your model does not achieve target performance.*

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimisation.ipynb`.

2. Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimising the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimise the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimisation.h5`.

## Step 4: Write a Report on the Neural Network Model

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

2. **Results:** Using bulleted lists and images to support your answers, address the following questions:

    * Data Preprocessing

        * What variable(s) are the target(s) for your model?
        
        * What variable(s) are the features for your model?

        *What variable(s) should be removed from the input data because they are neither targets nor features?
    
    * Compiling, Training, and Evaluating the Model

        * How many neurons, layers, and activation functions did you select for your neural network model, and why?

        * Were you able to achieve the target model performance?

        * What steps did you take in your attempts to increase model performance?

3. **Summary:** Summarise the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

# Requirements

## Preprocess the Data (30 points)

* Create a dataframe containing the `charity_data.csv` data , and identify the target and feature variables in the dataset (2 points)

* Drop the `EIN` and `NAME` columns (2 points)

* Determine the number of unique values in each column (2 points)

* For columns with more than 10 unique values, determine the number of data points for each unique value (4 points)

* Create a new value called `Other` that contains rare categorical variables (5 points)

* Create a feature array, `x`, and a target array, `y` by using the preprocessed data (5 points)

* Split the preprocessed data into training and testing datasets (5 points)

* Scale the data by using a *StandardScaler* that has been fitted to the training data (5 points)

## Compile, Train and Evaluate the Model (20 points)

* Create a neural network model with a defined number of input features and nodes for each layer (4 points)

* Create hidden layers and an output layer with appropriate activation functions (4 points)

* Check the structure of the model (2 points)

* Compile and train the model (4 points)

* Evaluate the model using the test data to determine the loss and accuracy (4 points)

* Export your results to an HDF5 file named `AlphabetSoupCharity.h5` (2 points)

## Optimise the Model (20 points)

* Repeat the preprocessing steps in a new Jupyter notebook (4 points)

* Create a new neural network model, implementing at least 3 model optimisation methods (15 points)

* Save and export your results to an HDF5 file named `AlphabetSoupCharity_Optimisation.h5` (1 point)

## Write a Report on the Neural Network Model (30 points)

* Write an analysis that includes a title and multiple sections, labeled with headers and subheaders (4 points)

* Format images in the report so that they display correction (2)

* Explain the purpose of the analysis (4)

* Answer all 6 questions in the results section (10)

* Summarise the overall results of your model (4)

* Describe how you could use a different model to solve the same problem, and explain why you would use that model (6)

# Resources

BCS Xpert Learning assistant

https://en.wikipedia.org/wiki/Layer_(deep_learning)#:~:text=A%20layer%20in%20a%20deep,it%20to%20the%20next%20layer.&text=(AlexNet%20image%20size%20should%20be,math%20will%20come%20out%20right.

https://www.analyticsvidhya.com/blog/2021/03/basics-of-neural-network/

https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/

https://stackoverflow.com/questions/35345191/what-is-a-layer-in-a-neural-network

https://stackoverflow.com/questions/41410317/why-do-we-have-multiple-layers-and-multiple-nodes-per-layer-in-a-neural-network

https://www.w3schools.com/python/python_ml_getting_started.asp

https://www.geeksforgeeks.org/introduction-deep-learning/

# Acknowledgments 

Dataset provided by provided by edX Boot Camps LLC.
