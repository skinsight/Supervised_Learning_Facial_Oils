###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['Price ($)','#Reviews']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 50))
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_yticklabels([0, 10, 20, 30, 40, 50, ">50"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (12.5,10))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                #ax[j//3, j%3].set_xlim((-0.1, 3.0))
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    gt_zero = importances > 0
    importances_gt_zero = importances[gt_zero]
    columns_gt_zero = X_train.columns.values[gt_zero]
    values_gt_zero = importances[gt_zero]
    indices = np.argsort(importances_gt_zero)[::-1]
    columns = columns_gt_zero[indices][:10]
    values = importances_gt_zero[indices][:10]
    len_range = len(columns)
    #print(columns)

    # Creat the plot
    fig = pl.figure(figsize = (7,7))
    pl.title("Normalized Weights For The Top Ten Predictive Features", fontsize = 16)
    pl.bar(np.arange(len_range), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(len_range) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(len_range), columns, rotation='vertical')
    pl.xlim((-0.5, len_range + 0.3))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  
    
def generate_ingredient_ratings_analysis_good(df, analyze_var, rating_threshold):
    
    analyze_var_threshold = rating_threshold #df[analyze_var].median()
    df_high = df[df[analyze_var] > analyze_var_threshold]
    df_high_orig = df_high
    ingredients_count = {}
    for ingredient_data in df_high_orig['Ingredients_Clean'].tolist():
        ingredient_list = ingredient_data.strip().split(',')
        for ingredient in ingredient_list:
            if ingredient in ingredients_count:
                ingredients_count[ingredient] = ingredients_count[ingredient] + 1
            else:
                ingredients_count[ingredient] = 1
    return ingredients_count
    
def generate_ingredient_ratings_analysis_bad(df, analyze_var, rating_threshold):
    
    analyze_var_threshold = rating_threshold
    df_high = df[df[analyze_var] < analyze_var_threshold]
    df_high_orig = df_high
    ingredients_count = {}
    for ingredient_data in df_high_orig['Ingredients_Clean'].tolist():
        ingredient_list = ingredient_data.strip().split(',')
        for ingredient in ingredient_list:
            if ingredient in ingredients_count:
                ingredients_count[ingredient] = ingredients_count[ingredient] + 1
            else:
                ingredients_count[ingredient] = 1
    return ingredients_count
    
def generate_ingredient_columns(df):
    df_new = df.copy()
    ingredients = {}
    for ingredient_data in df['Ingredients_Clean'].tolist():
        ingredient_list = ingredient_data.strip().split(',')
        for ingredient in ingredient_list:
            if ingredient not in ingredients:
                ingredients[ingredient] = 1

    for ingredient in ingredients:
        ingredient_column = []
        for ingredient_data in df['Ingredients_Clean'].tolist():
            ingredient_list = ingredient_data.strip().split(',')
            if ingredient in ingredient_data:
                ingredient_column.append(1)
            else:
                ingredient_column.append(0)
        df_new[ingredient] = ingredient_column
    return df_new

def generate_ingredients_list(df):
    ingredients_list = df['Ingredients'].tolist()
    ingredients_list_full = []
    for ingredients in ingredients_list:
        ing_list = ingredients.strip().split(',')
        for ing in ing_list:
            #import pdb; pdb.set_trace()
            ing = ing.strip()
            #if ing not in ingredients_list_full:
            ingredients_list_full.append(ing.strip())
    return ingredients_list_full

def generate_ingredients_clean(df, ingredient_map):
    ingredients_list = df['Ingredients'].tolist()
    ingredients_list_new = []
    for ingredients in ingredients_list:
        ing_list = ingredients.strip().split(',')
        ing_list_new = []
        for ing in ing_list:
            #import pdb; pdb.set_trace()
            ing = ing.strip()
            if ing in ingredient_map:
                ing_list_new.append(ingredient_map[ing].strip())
            else:
                ing_list_new.append(ing.strip())
        ing_list_newString = ",".join(ing_list_new)
        ingredients_list_new.append(ing_list_newString)
    df['Ingredients_Clean'] = ingredients_list_new
    return df

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: rating training set
       - X_test: features testing set
       - y_test: rating testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)    
    predictions_train = learner.predict(X_train[:])    
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the training samples which is y_train[:]
    results['acc_train'] = accuracy_score(y_train[:], predictions_train)    
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)  

    
    # TODO: Compute F-score on the training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:], predictions_train, beta = 0.5)
    
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
    
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

def evaluate_models(X_train, y_train, X_test, y_test):
    # TODO: Import the three supervised learning models from sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import svm
    
    
    
    # TODO: Initialize the three models
    clf_A = RandomForestClassifier(random_state = 100)
    # Tuned the parameters using Grid Search via cross validation
    clf_B = AdaBoostClassifier(n_estimators=400, random_state = 100, learning_rate=0.01)
    clf_C = svm.SVC(random_state = 100, kernel='linear')
    
    samples = len(y_train)
    
    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples, samples, samples]):
            results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)             
            print("ModelName: {}, F-score: {}, Accuracy: {}, train_time: {}, pred_time: {}, samples: {}".format(clf_name, results[clf_name][i]['f_test'], results[clf_name][i]['acc_test'], results[clf_name][i]['train_time'], results[clf_name][i]['pred_time'], samples))
    # Run metrics visualization for the three supervised learning models chosen
    return results