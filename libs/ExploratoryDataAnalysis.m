% This script output summary and other descriptive statistics about the
% data, in the form of figures and console prints.
function ExploratoryDataAnalysis (data, catColumns, numericCols)
    fprintf("Performing exploratory data analysis...\n");
    % Data Shape
    [rows, columns] = size(data);
    fprintf("The data contains %d observations with %d columns\n\n", rows, columns);
    fprintf("The data set contains %d numerical and %d categorical features.\n",...
        length(numericCols), length(catColumns));
    % show first 5 rows
    disp("Display the first 5 rows");
    head(data, 5)
    % summarize the data
    disp("Print summary of data table:");
    summary(data);
    
    % Univariate visualizations
    % plot gender demographics
    figure('Name', "Distribution of Demographics", 'pos',[10 10 800 600])
    subplot(3,2,1)
    pie(data.gender_of_respondent)
    title("Gender")
    % plot gender demographics
    subplot(3,2,2)
    histogram(data.age_of_respondent)
    title("Age")
    % plot location_type demographics
    subplot(3,2,3)
    histogram(data.marital_status)
    title("Marital status")
    % plot location_type demographics
    subplot(3,2,4)
    pie(data.location_type)
    title("Rural vs Urban")
    % plot country demographics
    subplot(3,2,5)
    pie(data.location_type)
    title("Demographics: Country")
    % plot relationship_with_head demographics
    subplot(3,2,6)
    histogram(data.relationship_with_head)
    title("Relationship with gead of house")
    % plot target
    figure('pos',[10 400 500 400])
    histogram(data.bank_account)
    title('Target Variable : Bank Account')
    ylabel('Number of Observations');
    
    % Bivariate visualizations of features vs target column
    figure('Name', "Distribution of Demographics", 'pos',[400 10 1200 800])
    featuredCols = {'education_level', 'job_type', 'age_of_respondent', 'country'};
    
    data_yes = data(data.bank_account == 'Yes', :);
    data_no = data(data.bank_account == 'No', :);
    for i = 1:numel(featuredCols)
        col_name = featuredCols{i};
        xLabel = Helper.normalizeLabel(col_name);
        subplot(2, 2, i)
        % Normalize the histograms so that all of the bar heights add to 1,
        % and use a uniform bin width.
        histogram(data_no.(col_name), 'FaceColor','b');
        hold on
        histogram(data_yes.(col_name), 'FaceColor','g');
        % label axis
        ylabel('No of observations');
        xlabel(xLabel);
        % add title and legend
        title(sprintf("%s vs bank_account", xLabel));
        legend('No', 'Yes', 'Location', 'Best');
    end
    
    % Correlation visulations
    numericColsFilter = ismember(data.Properties.VariableNames, numericCols);
    corrMatrix = corr(table2array(data(:, numericColsFilter)),'type','Pearson');
    figure('Name', "Relationship between numeric Attributes", 'pos',[200 200 500 400])
    imagesc(corrMatrix)
    colorbar; % enable colorbar
    colormap('parula'); % set the colorscheme
    
    n = length(numericCols);
      
    % replace underscore in labels with spaces
    colLabels = {Helper.normalizeLabel(numericCols{1}),...
        Helper.normalizeLabel(numericCols{2})};
        
    set(gca, 'XTick', 1:n); % center x-axis ticks on bins
    set(gca, 'YTick', 1:n); % center y-axis ticks on bins
    set(gca, 'XTickLabel', colLabels); % set x-axis labels
    set(gca, 'YTickLabel', colLabels); % set y-axis labels
    % set title
    title('Pearson correlation plot of numeric attributes', 'FontSize', 14);
    
    % Create scatter plots comparing a subset of the numeric variables.
    % Group the data according to the target class bank_account.
    figure('Name', "Relationship between numeric Attributes", 'pos',[200 10 500 400])
    color = lines(2);
    % standardize numerical data using zscore
    numericData = Helper.standardizeNumericAttributes(data(:, numericCols));
    % make a scatter plot matrix
    gplotmatrix(table2array(numericData), [], data.bank_account,...
        color,[],[],[],'grpbars',numericCols);
    title("Scatter plot matrix of numeric attributes", 'FontSize', 14);
end

