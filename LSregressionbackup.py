# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Least Squares Regression
# * Type of Data: Continuous
# * Basic Concept: Minimize distance from all data points by minimizing function: sum((Yi-Ypredictedi)^2) (also known as residual sum of squares, RSS)
# * Use case: 
#   * Continuation of the line of reasoning that first asks if there is a relationship, asks about the strength of the relationship, (James et al, 2013, p. 59-60)
# ## Statistical Model
# ### General Linear Model
# * Data = Model + Error
# * Error is random, with a mean of zero due to how the model is fitted to the data
# * Model of form y = mx + b (m is now called Beta1 and b Beta0)
# * We are trying to find the true model for the entire population based on the sample we have available
#   * The central idea is that as you increase the samples, the sample mean line of best fit will match the true line of best fit
# %% [markdown]
# <style> 
# div {line-height: 2}
# </style>
# <div class='bib'>
# <center>Resources</center>
# Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. (2013). An introduction to statistical learning: with applications in R. New York: Springer
# 
# https://github.com/JWarmenhoven/ISLR-python#readmehttps://github.com/JWarmenhoven/ISLR-python#readme
# 
# </div>

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Minimization of RSS with the variables Beta0 and Beta1
# This can be used by computign the partial derivatives of RSS in terms of Beta0 and then Beta1 in order to find the following formula for Beta0 and Beta1

def b0(x,y,beta1):
    y_mean = sum(y)/len(y)
    x_mean = sum(x)/len(x)
    
    b0 = y_mean - beta1*x_mean  
    return b0

def b1(x,y):
    x = np.array(x)
    y = np.array(y)

    y_mean = np.sum(y)/len(y)
    x_mean = np.sum(x)/len(x)

    b1 = np.sum((x-x_mean)*(y-y_mean))/np.sum(
        (x-x_mean)*(x-x_mean))
    return b1

# Import advertising dataset
# This dataset is used to show the relationship between advertising budget and sales
# This is a dataset of sample means

import pathlib

adv_df = pd.read_csv(pathlib.Path('.\datasets\Advertising.csv'))

b1_ = b1(adv_df.TV.values, adv_df.Sales.values)

b0_ = b0(adv_df.TV.values, adv_df.Sales.values, b1_)

print('b0',b0_)
print('b1',b1_)


# %%
# Evaluating the wether there is indeed a relationship



