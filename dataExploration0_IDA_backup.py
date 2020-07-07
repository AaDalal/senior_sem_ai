# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Exploratory Data Analysis 0: Initial Data Analysis
# 
# Initial data analysis is a subset of exploratory data analysis which focuses on making the data fit to be put into a model. This means dealing with non-existant values, normalizing as necescary, and completing other tasks as necescary by the final model to be used.
# 
# Sources:
# https://reader.elsevier.com/reader/sd/pii/S0022522315017948?token=E85E57F81B03A15524B9F114673CAF3F3F0FF45188AA953EB7FDD8195887A04325990D11A24383AC4424F669BB95EDAE
# 
# https://towardsdatascience.com/dealing-with-missing-data-17f8b5827664 
# 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/

# %%
import pandas as pd 
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import pathlib

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# %%
# Initializing data
vg_df = pd.read_csv(pathlib.Path('vgsales.csv'))


# %%

vg_df.dtypes
vg_df.describe()

InteractiveShell.ast_node_interactivity = "last_expr"

# %% [markdown]
# ### Check for duplicates

# %%
# Check for duplicates
duplicate_bool_ser = vg_df.duplicated(keep = False)

duplicate_bool_ser[duplicate_bool_ser == True]

# %% [markdown]
# ### Check for Nonexistant Values/Nones (grouped together as nulls) and other unknowns

# %%
# Plotting the locations of the NaNs by row

nan_locs = vg_df[vg_df.isnull().any(axis = 1)].index.tolist()

plt.bar(nan_locs, 1, width = 10)
plt.title("Null Values by Row")
plt.show()

# Binned by thousands

plt.hist(np.array(nan_locs), bins = 17, range = (0.0, 17000.0))
plt.title('Occurences of NaN in Dataframe binned by 1000s')
plt.xlabel('Row Number')
plt.ylabel('Number of Occurences within Each Bin of 1000 rows')
plt.show()

# Table of where the NaNs are located

NaN_count_col_df = pd.DataFrame()
for col in vg_df.columns:
    NaN_count_col_df[f'NaNs_in_{col}'] = [vg_df[col].isnull().values.sum()]
NaN_count_col_df


# %%
InteractiveShell.ast_node_interactivity = "all"

# Found synonyms of null, unknown, nonexistant, n/a, not any etc that might be present in a dataset

null_synonyms = ['unknown', 'untold', 'undetermined', 'undefined', 'hidden', 'indefinite', 'pending', 'inconclusive', 'unnamed', 'undesignated', 'insignificant', 'nonexistant', 'non-existant', 'missing', 'absent', 'unavailable', 'nonexistent', 'withdrawn', 'null', 'invalid', 'void', 'rescinded', 'repealed', 'blank', 'empty', 'canceled', 'revoked', 'rescinded', 'not any', 'n/a', 'None', 'nan', 'excluded',]

# Select columns with dtype 'object' and converting all strings to lowercase
obj_cols = ['Name', 'Platform', 'Genre', 'Publisher']

obj_vg_df = pd.DataFrame(dtype = 'object')
for col in obj_cols:
    obj_vg_df[col] = vg_df[col].str.lower()

# Recording where the word occus

for word in null_synonyms:
    for col in obj_cols:
        if True in (obj_vg_df[col] == word).values:
            print('Word: ',word)
            print('in Column: ',col)
            np.array(obj_vg_df.index[obj_vg_df[col] == word].tolist())
            # The output list tells us that the only occurence of a synonym of null was 'unknown' in the 'publisher column'

# Plotting the locations of the 'unknown's by row

unknown_indices = obj_vg_df.index[obj_vg_df['Publisher'] == 'unknown'].tolist()

plt.bar(unknown_indices, 1, width = 10)
plt.title("Unknown Values by Row")
plt.show()

# Binned by thousands

plt.hist(np.array(unknown_indices), bins = 17, range = (0.0, 17000.0))
plt.title('Occurences of "unknown" in "publisher" column binned by 1000s')
plt.xlabel('Row Number')
plt.ylabel('Number of Occurences within Each Bin of 1000 rows')
plt.show()

# %% [markdown]
# ### How to deal with these NaN and unknown values?
# 
# Simple options:
# * Delete the features with NaN and unknown values from the dataset entirely
# * Delete rows with those features missing
# * Delete the chunk of rows with those features missing
# * Turn NaN/unknown into a category
#   * Eg., if options for publishers are 'Nintendo', 'Sega' etc add a new option of 'unknown'
# * Replace with mean, median or mode
# 
# From Kang (2013):
# * 
