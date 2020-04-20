%  HyperParameterTuning uses the 'OptimizeHyperparameters' name-value pair 
%  argument to find optimal parameters automatically using Bayesian optimization. 
%  The optimal values obtained by using the auto search will then
%  be used to train the best model.
function [nbTuned, trTuned] = HyperParameterTuning(X, y, classNames)
    fprintf("Starting hyper-parameter tunning...\n");
    % standardized numerical data using zscore
    X = Helper.standardizeNumericAttributes(X);
    %Create independent training and test sets of data. Use 75% of the data
    %for a training set by calling cvpartition using the holdout option.
    [X_train, y_train, X_test, y_test] = Helper.train_test_split(X,y, 0.25);
    % train the classifiers and store it in a cell array
    rng(5); % For reproducibility
    % To speed up the process, specify 'UseParallel' as true to run
    % Bayesian optimization in parallel.
    % Use 'expected-improvement-plus' as acquisition function for reproducibility
    hypopts = struct('UseParallel', true,...
        'AcquisitionFunctionName', 'expected-improvement-plus');
    try
        % Start a parallel pool.
        poolobj = gcp;
    catch
        % disable parallel run if Parallel Computing Toolbox is not
        % available
        hypopts.UseParallel = false;
    end
    % Naive Bayes
    nbTuned = fitcnb(X_train,y_train, 'ClassNames', classNames,...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions', hypopts);
    
    % Train random forest with
    t = templateTree;
    trTuned = fitcensemble(X_train,y_train, 'ClassNames', classNames,...
        'Learners', t, 'Method', 'Bag',...
        'OptimizeHyperparameters', {'NumLearningCycles','MaxNumSplits'},...
        'HyperparameterOptimizationOptions', hypopts);
    
    % Plot Minimum Objective Curves
    figure('Name', 'Hyperparameter Tuning Performance Comparison',...
        'pos', [10 10 1200 800])
    subplot(3,2,[1,3])
    hold on
    mdls = {nbTuned, trTuned};
    N = length(mdls);
    for i = 1:N
        % return the optimization results
        results = mdls{i}.HyperparameterOptimizationResults;
        % plot Minimum Objective Curve
        plot(results.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
    end
    grid on
    names = {'Naive Bayes', 'Random Forest'};
    legend(names{1}, names{2},'Location','northeast')
    title('Plot Minimum Objective Curves - Bayesian Optimization')
    xlabel('Number of Iterations')
    ylabel('Minimum Objective Value')
    
    % Check Performance with Test Set
    % Find the predicted labels and the score values of the test data set.
    % define a prediction function that takes a trained model and make
    % predictions
    predictFn = @(model) predict(model, X_test);
    % define a function that takes posterior probabilities of model 
    % predictions and compute ROC parameters. 
    rocCurveFn = @(scores) perfcurve(y_test, scores(:,1), 'Yes');
    
    labels = cell(N,1);
    scores = cell(N,1);
    
    % Obtain the most likely class for each test observation by using the
    % predict function of each model.
    for i = 1:N
        [labels{i}, scores{i}] = predictFn(mdls{i});
    end
    
    % Compute the standard ROC curve using the scores from the naive Bayes
    % and random forest classification.
    [Xnb,Ynb,~,AUCnb, optNB] = rocCurveFn(scores{1});
    [Xtr,Ytr,~,AUCtr, optTr] = rocCurveFn(scores{2});
    % Plot the Naive Bayes ROC curves on the graph.
    subplot(3, 2, [2, 4])
    plot(Xnb,Ynb, 'b')
    hold on
    % plot Optimal operating point of Naive Bayes
    plot(optNB(1), optNB(2),'bo')
    % Plot the Random forest ROC curves on the same graph.
    plot(Xtr,Ytr, 'r')
    % plot Optimal operating point of Random forest
    plot(optTr(1), optTr(2),'ro')
    % write AUC on plot
    text(0.5,0.5,strcat('Naive Bayes AUC = ',num2str(AUCnb)),...
        'EdgeColor','b', 'HorizontalAlignment','center')
    text(0.5,0.55,strcat('Random Forest AUC = ',num2str(AUCtr)),...
        'EdgeColor','r', 'HorizontalAlignment','center')
    legend('Naive Bayes', 'Naive Bayes OPTROCPT',...
        'Random Forest', 'Random Forest OPTROCPT',...
        'Location','Best')
    grid on
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves for Naive Bayes and Random Forest')
    hold off
    
    % Plot Confusion Matrix of tuned models
    for i = 1:N
        subplot(3,2,4+i)
        Helper.plotConfusionMatrix(y_test, labels{i}, names{i});
    end
    
    % return models trained models
    nbTuned = mdls{1};
    trTuned = mdls{2};
end