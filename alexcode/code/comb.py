import pandas as pd

s1= pd.read_csv('submission_1.csv')
s2= pd.read_csv('submission_2.csv')

for i in range(s1.shape[0]):
    s1.iloc[i, 1:601] = 2 * (s1.iloc[i, 1:601] * s2.iloc[i, 1:601])/ (s1.iloc[i, 1:601] + s2.iloc[i, 1:601])

s1.to_csv('comb.csv', index = False)