

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


df= pd.read_csv('HappinessReport2020.csv')
df.head()

df.columns

col_rename = {'Country name':'Country', 'Regional indicator':'Region', 'Ladder score': 'Ladder',
                  'Standard error of ladder score':'Standard Error', 'Logged GDP per capita':'Logged GDPPC',
                  'Social support':'Social Support', 'Healthy life expectancy':'Life Expectancy',
                  'Freedom to make life choices':'Freedom', 'Perceptions of corruption': 'Corruption'}


df.rename(columns = col_rename, inplace = True)
df.head()


df_drop = df.drop(['Standard Error', 'upperwhisker', 'upperwhisker', 'lowerwhisker', 'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption'], axis=1)



plt.rcParams['figure.figsize'] = (12,8)
sns.heatmap(df_drop.corr(), cmap = 'copper', annot = True)
plt.show()
"""
Ladder có mối tương quan rất cao với GDP , Social support và Healthy life expectancy.
Ngược lại, có mối quan hệ rất thấp với Perceptions of corruption.
"""

fig = plt.figure(figsize = (18, 14))
ax = plt.axes()
"""
Ta thấy được phân bổ  cáu quosc gia theo khu vực và mức độ hạnh phúc khác nhau giữa các khu vực
"""
countplot = sns.countplot('Region', data = df, saturation = 0.8, palette = 'tab10')
countplot.set_xticklabels(countplot.get_xticklabels(), rotation = 90)
countplot.set_title("Countplot by Region", y = 1.05);



feature_cols = ['Logged GDPPC', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
df = pd.concat([df['Ladder'], df[feature_cols]], axis = 1)

fig = plt.figure(figsize = (13, 10))
plt.style.use('seaborn-white')

plt.matshow(df.corr(), fignum = fig.number, cmap = 'viridis')
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize = 24, y = 1.2);
"""
Xem xét các mối quan hệ giữa mỗi trong số sáu giá trị được đo lường
"""


fig, axes = plt.subplots(2, 3, figsize = (20, 12))

for i, ax in enumerate(axes.flat):
    ax.plot(df['Rank'], df[feature_cols[i]], color = 'red')
    ax.set_title(feature_cols[i] + ' by Rank', fontsize = 18)
    ax.set_xlim(153, 1)
    ax.axis('tight')
plt.show()


pairplot = sns.pairplot(df, hue = 'Quartile', vars = feature_cols, corner = False)
pairplot.fig.suptitle("Pairplot of the 6 Happiness Metrics", fontsize = 24, y = 1.05);
plt.show()






