# Introduction

The Readme file contains instructions and comments on the deliverables of the course work developed by Philip Adzanoukpe and Addai-Marnu Daniel. Before moving on, please take a moment to read it.

# Matlab Version

Matlab R2019a Update 6 (9.6.0.1214997) was used to develop all the work.

# Project Structure


```bash
.
├── README.txt
├── data
│   └── financial_inclusion.csv
├── libs
│   ├── BestModelRF.m
│   ├── DecisionBoundary.m
│   ├── ExploratoryDataAnalysis.m
│   ├── Helper.m
│   ├── HyperParameterTuning.m
│   ├── PerformanceComparison.m
│   ├── PreProcessData.m
│   └── PredictorImportance.m
└── main.m
```

# Scripts

In this section, I explain how the scripts can be run.

To run the program, run the main.m file. It serves as an entry point to run all scripts. 

Running the main.m file asks the user to select a number that matches the script to run. Type 0 to leave the programme. 

The main entry point loads the data from the data directory and passes it as an argument to the selected function.


Definition of each scripts:

	- main.m : Main interface for running all scripts from one command.

	- ExploratoryDataAnalysis.m : This script output summary and other descriptive statistics about the data, in the form of figures and console prints.

	- PredictorImportance.m : This script runs to compute predictor importance rank using decision trees and visualize the features in a bar graph.

	- DecisionBoundary.m : This script visualizes the decision boundaries of Naive Bayes and Random Forest in 2D by training the models on the numerical attributes. The goal here is to show the difference between the shapes.

	- PerformanceComparison.m : This script shows the difference in predictive performance and time performance between the two algorithms. It uses a simple train-test cross-validation.
    
    - HyperParameterTuning.m : This script performs auto optimization of the hyper-parameters using Bayesian optimization to obtain the optimal values for each models.

	- BestModelRF.m : This scripts contains a function to run the Random Forest with the hyper-parameters and shows the difference from using the default options.
    
    - Helper.m : This contains a helper class with static methods which shared among all the scripts.
