function [cleanData, X_data, y_data] =  PreProcessData (rawData, catColumns, targetCol) 
    fprintf("Preprocessing data...\n");

    % convert categorical columns to categorical data type
    cleanData = convertvars(rawData, catColumns, 'categorical');
    cleanData = convertvars(cleanData, targetCol, 'categorical');
    % drop uniqueid column as it will not be used in the analysis
    cleanData = removevars(cleanData, "uniqueid");
    % To simplify the table, we move uniqueid and bank_account to the
    % begining and end of the table respectively
    cleanData = movevars(cleanData, "bank_account", "After", "job_type");
    % check if there are missing values
    missingValues = any(ismissing(cleanData));
    if missingValues
        fprintf("There are %d missing values in the data set.", length(missingValues));
    else
        fprintf("There are no missing values in the data set.");
    end
    fprintf("\n\n"); 
    
    predictorFilter = ismember(cleanData.Properties.VariableNames, targetCol);
    predictorNames = cleanData.Properties.VariableNames(~predictorFilter);
    % isCategoricalPredictor = ismember(predictorNames, catColumns);

    X_data = cleanData(:, predictorNames);
    y_data = cleanData.bank_account;
end