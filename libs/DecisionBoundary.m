% Comparison of the shape of Decision Boundaries between Gaussian Naive
% Bayes and Random forest classifier
% We fit the classifiers using two chosen continuous attributes to be able
% to visualize in 2D and 3D

function DecisionBoundary(data, classNames)
    fprintf("Computing decision boundaries...\n");
    % Undersampling the majority class 'No' to get rid of the imbalance class
    % problem
    data = Helper.undersampleMajorityClass(data);
    cols = {'age_of_respondent', 'household_size'}; % selected numerical columns
    X = data(:, cols);
    y = data.bank_account;
    
    % standardized numerical data using zscore
    X = Helper.standardizeNumericAttributes(X);
    
    % Define a grid of values in the observed predictor space. Predict the
    % posterior probabilities for each instance in the grid.
    x1Max = max(X.(cols{1})); x1Min = min(X.(cols{1}));
    x2Max = max(X.(cols{2})); x2Min = min(X.(cols{2}));
    step = 0.01;
    [x1Grid,x2Grid] = meshgrid(x1Min:step:x1Max, x2Min:step:x2Max);
    XGrid = [x1Grid(:),x2Grid(:)];
    
    classifier_name = {'Gaussian Naive Bayes','Random Forest'};
    % Train a naive Bayes classifier.
    mdlNB = Helper.fitNaiveBayes(X,y, classNames);
    % Train random forest classifier with tuned parameters.
    mdlRF = Helper.fitRandomForest(X,y, classNames);
    % Visualize the scatterplot of Age of Respondent vs Household size 
    % colored by class labels
    figure('Name', 'Decision Surface for each classifier',...
        'pos', [300 200 1200 600])
    subplot(2, 2, [1, 3])
    gscatter(data.age_of_respondent, data.household_size,...
        data.bank_account, [0 0.4470 0.7410; 0.8500 0.3250 0.0980], 'xo')
    title('Plot of Age of Respondent vs Household size with target class')
    % label axis.
    xLabel = Helper.normalizeLabel(cols{1});
    yLabel = Helper.normalizeLabel(cols{2});
    xlabel(xLabel);
    ylabel(yLabel);
    hold on
    
    % predict the labels and posterior probabilities for each observation
    % using all classifiers.
    % make predictions for naive bayes
    [predNB, posteriorNB] = predict(mdlNB, XGrid);
    % make predictions for random forest
    [predRF, posteriorRF] = predict(mdlRF, XGrid);
    
    % DECISION SURFACE
    % Visualize the Decision Surface for each classifier 
    % Gaussian Naive Bayes
    subplot(2, 2, 2)
    gscatter(x1Grid(:), x2Grid(:), predNB, 'rb');
    title(classifier_name{1})
    xlabel(xLabel);
    ylabel(yLabel);
    % Random Forest
    subplot(2, 2, 4)
    gscatter(x1Grid(:), x2Grid(:), predRF, 'rb');
    title(classifier_name{2})
    xlabel(xLabel);
    ylabel(yLabel);
    
    % Plot the posterior probability regions for each classifier.
    sz = size(x1Grid);
    % Gaussian Naive bayes
    % Visualize the Probability Decision Boundary in 2D of Gaussian Naive
    % Bayes.
    figure('Name', 'PREDICTED PROBABILITIES OF CLASSIFICATION MODELS',...
        'pos', [10 10 1200 600])
    subplot(2, 2, 1)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorNB(:,1), sz),...
        'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorNB(:,2), sz),...
        'EdgeColor', 'none')
    colorbar
    view(2)
    title('Probability Decision Boundary - NAIVE BAYES')
    xlabel(xLabel);
    ylabel(yLabel);
    hold off
    % Visualize the Posterior Probability Distribution of Target
    % Classes in 3D for Gaussian Naive bayes
    subplot(2, 2, 2)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorNB(:,1), sz),...
        'FaceColor', 'red', 'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorNB(:,2), sz),...
        'FaceColor', 'blue', 'EdgeColor', 'none')
    alpha(0.2)
    view(3)
    title('Posterior Probability Distribution of each Class - NAIVE BAYES')
    xlabel(xLabel);
    ylabel(yLabel);
    legend({'No', 'Yes'})
    hold off
    % Random Forest
    % Visualize the Probability Decision Boundary in 2D for Random Forest
    subplot(2, 2, 3)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorRF(:,1), sz),...
        'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorRF(:,2), sz),...
        'EdgeColor', 'none')
    colorbar
    view(2)
    title('Probability Decision Boundary - RANDOM FOREST')
    xlabel(xLabel);
    ylabel(yLabel);
    hold off
    % Visualize the Posterior Probability Distribution of Target
    % Classes in 3D for Random Forest.
    subplot(2, 2, 4)
    hold on
    surf(x1Grid, x2Grid, reshape(posteriorRF(:,1), sz),...
        'FaceColor', 'red', 'EdgeColor', 'none')
    surf(x1Grid, x2Grid, reshape(posteriorRF(:,2), sz),...
        'FaceColor', 'blue', 'EdgeColor', 'none')
    alpha(0.2)
    view(3)
    title('Posterior Probability Distribution of each Class - RANDOM FOREST')
    xlabel(xLabel);
    ylabel(yLabel);
    legend({'No', 'Yes'})
    hold off
    
end