% Compare classification performance of Naive Bayes classifier and Ensemble
% Classification tree using default parameters.
function PerformanceComparison(X, y, classNames)
    fprintf("Running performance comparision...\n");
    % split data into training and tests using 75% for training and 25% for
    % validation
    [X_train, y_train, X_test, y_test] = Helper.train_test_split(X,y, 0.25);
    
    % standardized numerical data using zscore
    X_train = Helper.standardizeNumericAttributes(X_train);
    X_test = Helper.standardizeNumericAttributes(X_test);
    % define a prediction function that takes a trained model and make
    % predictions
    predictFn = @(model) predict(model, X_test);
    % define a function that takes posterior probabilities of model 
    % predictions and compute ROC parameters. 
    rocFn = @(scores) perfcurve(y_test, scores(:,1), 'Yes');
    
    % Train Gaussian Naive Bayes
    mdlNBTrainFn = @() Helper.fitNaiveBayes(X_train, y_train, classNames);
    mdlNB = mdlNBTrainFn();
    % Compute the posterior probabilities (scores).
    mdlNBPredictFn = @() predictFn(mdlNB);
    [nbLabels, posteriorNB] = mdlNBPredictFn();
    % Compute the standard ROC curve using the scores from the naive Bayes
    % classification.
    [Xnb,Ynb,~,AUCnb, optNB] = rocFn(posteriorNB);
    
    % Train Random forest with 100 decision trees
    mdlTreeTrainFn = @() Helper.fitRandomForest(X_train, y_train,...
        classNames);
    mdlTree = mdlTreeTrainFn();
    mdlTreePredictFn = @() predictFn(mdlTree);
    % Compute the posterior probabilities (scores).
    [trLabels,posteriorTr] = mdlTreePredictFn();
    [Xtr,Ytr,~,AUCtr, optTr] = rocFn(posteriorTr);
    
    
    % track model timing
    timeNBfit = timeit(mdlNBTrainFn);
    timeNBpred = timeit(mdlNBPredictFn);
    timeRFfit = timeit(mdlTreeTrainFn);
    timeRFpred = timeit(mdlTreePredictFn);
    
    % Printing Performance
    % Prediction Performance
    fprintf('AUC Prediction Performance -----------\n')
    fprintf('Naive Bayes         : %.3f\n', AUCnb)
    fprintf('Random Forest : %.3f\n\n', AUCtr)
        % Time Performance
    fprintf('Time Performance (seconds) -----------\n')
    fprintf('      Training Time Performance ------\n')
    fprintf('Naive Bayes         : %.3f\n', timeNBfit)
    fprintf('Random Forest : %.3f\n', timeRFfit)
    fprintf('      Prediction Time Performance ----\n')
    fprintf('Naive Bayes         : %.3f\n', timeNBpred)
    fprintf('Random Forest : %.3f\n', timeRFpred)
    
    % Plot the ROC curves on the same graph.
    figure('Name', 'Performance comparision of classifiers',...
        'pos', [100 100 1200 600]);
    subplot(2, 2, [1, 3])
    plot(Xnb,Ynb, 'b')
    hold on
    plot(Xtr,Ytr, 'r')
    % plot Optimal operating point of the ROC curve
    plot(optNB(1), optNB(2),'bo')
    plot(optTr(1), optTr(2),'ro')
    grid on
    % write AUC on plot
    text(0.3,0.35,strcat('Naive bayes AUC = ',num2str(AUCnb)),'EdgeColor','b')
    text(0.3,0.30,strcat('Random Forest AUC = ',num2str(AUCtr)),'EdgeColor','r')
    legend('Naive Bayes', 'Random Forest', 'Naive Bayes OPTROCPT',...
         'Random Forest OPTROCPT', 'Location','Best')
    
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves for Naive Bayes and Random Forest')
    hold off
    
    % plot confusion metrics
    % visualize confusion metrics for Naive bayes classifier
    subplot(2, 2, 2)
    Helper.plotConfusionMatrix(y_test, nbLabels, 'Naive Bayes');
    
    % visualize confusion metrics for Naive bayes classifier
    subplot(2, 2, 4)
    Helper.plotConfusionMatrix(y_test, trLabels, 'Random Forest');
end