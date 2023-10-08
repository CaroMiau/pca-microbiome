# The following code references the code available on:
# https://github.com/StatQuest/pca_demo/blob/master/pca_demo.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os

# Function to extract data from an Excel file
def extract_data_from_excel(file_path, file_name):
    print("querying " + file_name)
    df = pd.read_excel(file_path, sheet_name = "Especie")
    # Add a new column for the file name
    df['File'] = file_name
    return df

# Function to concatenate data from multiple Excel files
def concatenate_excel_data_from_folder(folder_path):
    combined_df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx') or filename.endswith('.xls') or filename.endswith('gsheet'):
            file_path = os.path.join(folder_path, filename)
            df = extract_data_from_excel(file_path, filename)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

# Specify the folder containing the Excel files
folder_path = '/content/drive/MyDrive/Carol/data'

# Call the function to concatenate data from Excel files to a DataFrame
combined_data_df = concatenate_excel_data_from_folder(folder_path)

# Cleaning merged df
df = combined_data_df[[100, "root", "File"]].copy()
df.rename(columns = {100: "percentage", "root": "microbiome"}, inplace = True)
df["microbiome"] = df["microbiome"].str.strip()
df['File'] = df['File'].str.replace('.xlsx', '')
df['point'] = df['File'].str.split().str[0]
df['region'] = df['File'].str.split().str[1]
df.drop(columns=["File"], inplace = True)

# Pivot_table to transform from long to wide dataframe
df_wide = df.pivot_table(index='microbiome', columns='point', values='percentage', aggfunc='sum')
df_wide = df_wide.dropna(axis=0, how='any')

#########################
#
# Perform PCA on the data
#
#########################
# First center and scale the data

scaled_data = preprocessing.scale(df_wide.T)
pca = PCA()
pca.fit(scaled_data)

pca_data = pca.transform(scaled_data)

#########################
#
# Draw a scree plot and a PCA plot
#
#########################
 
#The following code constructs the Scree plot

per_var = np.round(pca.explained_variance_ratio_* 100, decimals = 1)
labels = ['PC' + str (x) for x in range (1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, index = df_wide.columns, columns = labels)

colors = ['r', 'r', 'g', 'g', 'b', 'b', 'c', 'c', 'm', 'm',]

plt.scatter(pca_df.PC1, pca_df.PC2, c=colors, alpha=0.7)
plt.title("PCA graph")
plt.xlabel("PC1 - {0}%".format(per_var[0]))
plt.ylabel("PC2 - {0}%".format(per_var[1]))

for sample in pca_df.index:
  plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
plt.show()


#########################
#
# Determine which genes had the biggest influence on PC1
#
#########################
 
## get the name of the top 10 measurements (genes) that contribute
## most to pc1.
## first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index = df_wide.index )
## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values
## print the gene names and their scores (and +/- sign)
loading_scores[top_10_genes]