% Main entry point.

% Clear workspace and Command window
close all; clc;
% add libs folder to path
addpath(genpath("libs"));
% data source https://fsdkenya.org/publication/finaccess2019/
dataPath = sprintf('%s/data/financial_inclusion.csv', pwd);
rawData = readtable(dataPath);
% define categorical column parameters
catColumns = {
    'country', 'location_type', 'year',...
    'cellphone_access', 'gender_of_respondent', 'job_type',...
    'relationship_with_head', 'marital_status', 'education_level'
};
% define numerical column parameters
numericCols = {'household_size', 'age_of_respondent'};
% define the target column
targetCol = 'bank_account';
% define target labels
classNames = categorical({'Yes', 'No'});
%Pre process data by converting all categorical columns to categorical data
%type and reposition the target column to the end of the table.
[cleanData, X_data, y_data] = PreProcessData(rawData, catColumns, targetCol);
%% Main entry points to run scripts

section = 1;
while section ~= 0
    %close all; clear all;
    fprintf('\nPlease, type a number between 1 and 6 to run the related script, or 0 to exit the program\n\n')
    fprintf('>> 1 : Exploratory Data Analysis\n')
    fprintf('>> 2 : Predictor Importance\n')
    fprintf('>> 3 : Decision Boundaries: Naive Bayes vs Random Forest\n')
    fprintf('>> 4 : Performance of Models: Naive Bayes vs Random Forest\n')
    fprintf('>> 5 : Hyperparameter Tuning (warning: It may take a long time to complete.) \n')
    fprintf('>> 6 : Best Model: Random Forest\n')
    fprintf('Type 0 to exit the program ...\n\n')

    section = input('Enter a number: ');
    
    switch section
        case 1
            % Exploratory Data Analysis
            ExploratoryDataAnalysis(cleanData, catColumns, numericCols);
            pause(3)
        case 2
            % Determine predictor importance using decision tree
            PredictorImportance(cleanData, targetCol);
            pause(3)
        case 3
            % Decision Boundaries: Naive Bayes vs Random Forest
            DecisionBoundary(cleanData, classNames);
            pause(3)
        case 4
            % Performance of Models: Naive Bayes vs Random Forest
            PerformanceComparison(X_data, y_data, classNames);
            pause(3)
        case 5
            % Hyperparameter Tuning of Models
            [nb, rf] = HyperParameterTuning(X_data, y_data, classNames)
            pause(3)
        case 6
            % Comparison of Convergence Rate of the Generalization Error
            bestMdl = BestModelRF(X_data, y_data, classNames)
            pause(3)
        case 0
            continue
        otherwise
            % if number is invalid
            fprintf('\nPlease pick a viable number between 1 and 6.\n')
            fprintf('Type 0 to exit the program ...\n\n')
            pause(1)
    end
end

% Clear workspace, console
close all; clc;
fprintf('Program exited.\n\n')