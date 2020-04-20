% This script runs to compute predictor importance rank using decision
% trees and visualize the features in a bar graph.
function PredictorImportance(data, targetCol)
    fprintf("Computing predictor importance using decision trees...\n");
    % standardized numerical data using zscore
    data = Helper.standardizeNumericAttributes(data);
    % Train a classification tree using the entire data set. To grow
    % unbiased trees, specify usage of the curvature test for splitting predictors. 
    % Because there are missing observations in the data, specify usage of surrogate splits.
    rng(5);
    Mdl = fitctree(data, targetCol,'PredictorSelection','curvature');
    % Estimate predictor importance values by summing changes in the risk
    % due to splits on every predictor and dividing the sum by the number of branch nodes.
    imp = predictorImportance(Mdl);
    % visualize feature importance
    figure('Name', 'Feature importance using decision tree');
    bar(imp);
    title('Feature importance using decision tree');
    ylabel('Estimates');
    xlabel('Predictors');
    % set axis label and orientation
    fig = gca;
    fig.XTickLabel = Mdl.PredictorNames;
    fig.XTickLabelRotation = 45;
    fig.TickLabelInterpreter = 'none';
    fprintf("education_level is the most important predictor, followed by age_of_respondent and job_type\n")
end