function [mdlRf] = BestModelRF(X, y, classNames)
    fprintf("Running best model (Random Forest)...\n");
    % standardized numerical data using zscore
    X = Helper.standardizeNumericAttributes(X);
    %Create independent training and test sets of data. Use 75% of the data
    %for a training set by calling cvpartition using the holdout option.
    [X_train, y_train, X_test, y_test] = Helper.train_test_split(X,y, 0.25);
    
    % Create Random Foreset classifier of 19 trees and 184 MaxNumSplits
    % obtained after Hyperparameter tuning
    t = templateTree('Reproducible',true, 'MaxNumSplits', 184);  % For reproducibility
    mdlRf = Helper.fitRandomForestBest(X_train, y_train, classNames);
    % Generate a 10-fold cross-validated Random Foreset.
    cvmRF = fitcensemble(X,y,'ClassNames', classNames,...
        'Method','Bag',...
        'NumLearningCycles',19,...
        'Kfold',10,...
        'Learners',t...
    );
    % Train random forest with selected important features
    features = {'education_level', 'age_of_respondent', 'job_type',...
        'household_size'};
    smRF = Helper.fitRandomForestBest(X_train(:,features),y_train,classNames);
    
    % Plot the loss (misclassification) of the test data as a function of
    % the number of trained trees in the ensemble.
    figure('Name', 'Misclassification Rate - Random Forest',...
        'pos', [100, 100, 640, 480]);
    plot(loss(mdlRf,X_test,y_test,'mode','cumulative'))
    hold on
    % Examine the cross-validation loss as a function of the number of
    % trees in the ensemble.
    plot(kfoldLoss(cvmRF,'mode','cumulative'),'r.')
    % Visualize the loss curve for selected features
    plot(loss(smRF,X_test(:,features),y_test,'mode','cumulative'))
    % Generate the loss curve for out-of-bag estimates, and plot it along
    % with the other curves.
    plot(oobLoss(mdlRf,'mode','cumulative'),'k--')
    hold off
    xlabel('Number of trees')
    ylabel('Classification error')
    title('Generalization error of k-fold vs holdout cross-validation')
    legend('Holdout Test','10-fold test', 'Selected features',...
            'Out of bag', 'Location','NE')
    
    % visualize performance
    figure('Name', 'Performance metrics - Random Forest',...
        'pos', [400, 300, 640, 480]);
    [labels, scores] = predict(mdlRf, X_test);
    % Compute the ROC curve parameters
    [Xc,yc,~,AUC, OPTROCPT] = perfcurve(y_test, scores(:,1), 'Yes');
    plot(Xc,yc, 'b', 'LineWidth',2)
    hold on
    area(Xc, yc, 'LineStyle','none',...
        'FaceColor',[0.796875 0.88671875 0.9453125],...
        'EdgeColor','none');
    % plot Optimal operating point of the ROC curve
    plot(OPTROCPT(1), OPTROCPT(2),...
        'MarkerFaceColor',[0.85 0.325 0.098], 'MarkerSize',8,...
        'Marker','o','Color','none')
    grid on
    % write AUC on plot
    text(0.5,0.5, strcat('AUC = ',num2str(AUC)),...
        'HorizontalAlignment','center')
    text(0.5,0.55, 'Positive class: Yes ','HorizontalAlignment','center')
    % label axis
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves of the best model (Random Forest)')
    legend('ROC curve', 'Area under curve(AUC)',...
        'Current classifier', 'Location', 'SE')
    hold off
    
    % plot confusion metrics
    % visualize confusion metrics for the best model
    figure('Name', 'Performance metrics - Random Forest',...
        'pos', [600, 100, 600, 400]);
    Helper.plotConfusionMatrix(y_test, labels, 'Random Forest'); 
end

