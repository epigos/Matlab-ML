% This contains a helper class with static methods which shared among all
% the script.
classdef Helper

    methods(Static)
        function [X_train, y_train, X_test, y_test] = train_test_split(X, y, holdout)
            % Split training data into train and test using holdout
            % cross-validation method.
            rng("default") % For reproducibility
            cv = cvpartition(y,"Holdout",holdout);

            X_train = X(training(cv),:);
            y_train = y(training(cv));
            X_test = X(test(cv),:);
            y_test = y(test(cv));
        end
        
        function [distNames] = distributionNames(isCategoricalPredictor)
            distNames =  repmat({'Normal'}, 1, length(isCategoricalPredictor));
            distNames(isCategoricalPredictor) = {'mvmn'};
        end
        
        function [data] = undersampleMajorityClass(data)
           % Undersampling the majority class 'No' to get rid of the
           % imbalance class problem.
           data_yes = data(data.bank_account == 'Yes', :);
           rows = size(data_yes, 1);
           rng(5) % for reproducibility of next line
           data_no = datasample(data(data.bank_account == 'No', :), rows + 1000, 'Replace', false);
           data = vertcat(data_yes, data_no);
           % Shuffle data
           m = size(data, 1);
           data = data(randperm(m), :);
        end
        
        function [mdl] = fitRandomForest(X, y, classNames)
            % Train a Random forest classifier
            rng("default") % For reproducibility
            t = templateTree('Reproducible',true);
            mdl = fitcensemble(X,y, 'ClassNames', classNames,...
                'Learners', t,...
                'Method', 'Bag');
        end
        
        function [mdl] = fitRandomForestBest(X, y, classNames)
            % Create Random Foreset classifier of 19 trees and 184 MaxNumSplits
            % obtained after Hyperparameter tuning
            rng("default") % For reproducibility
            t = templateTree('Reproducible',true, 'MaxNumSplits', 184);
            mdl = fitcensemble(X,y, 'ClassNames', classNames,...
                'Learners', t, 'NumLearningCycles', 19,...
                'Method', 'Bag');
        end
        
        function [mdl] = fitNaiveBayes(X, y, classNames)
            % Train a Gaussian naive bayes classofier
            mdl = fitcnb(X,y, 'ClassNames', classNames);
        end
        
        function [X] = standardizeNumericAttributes(X)
            % standardized numerical data using zscore
            X.age_of_respondent = zscore(X.age_of_respondent);
            X.household_size = zscore(X.household_size);
        end
        
        function [yes, no] = priorProbabilities(y)
            % Obtain prior probabilites from the sample
            N = size(y, 1);
            data_yes = y(y == 'Yes', :);
            data_no = y(y == 'No', :);
            yes = size(data_yes, 1) / N;
            no = size(data_no,1) / N;
        end
        
        function [cm] = plotConfusionMatrix(y_test, labels, model_name)
            % Plot conusion matrix for model predictions
            cm = confusionchart(y_test, labels);
            txt = sprintf("Confusion matrix for %s", model_name);
            title(txt);
            disp(txt);
            cm.NormalizedValues
        end
        
        function [label] = normalizeLabel(colName)
            % Function to replace underscore with spaces in column names
            % for labeling axis in charts.
           label = strrep(colName, '_', ' ');
        end
    end

end


