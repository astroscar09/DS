import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split
from scipy.stats import f_oneway

#######
# 1-sample t-test for the mean


#######
# 1-sample t-test for the proportion (qualitative data)


#######
# 2-sample independent test for the mean

#######
# 2-sample independent test for the proportion

#######
# paired sample test (the same sample) [ex: before and after treatment on the same sample]

#######
# Regression test 

#######
# chi2 test test relationship between two categorical variables 
# (similar to regression but with qualitiative variables)

#######
# one way ANOVA Test is an N-sample independent test 

#Comparing sample to the populations
# stats.t.ppf(q = 0.025, 
#             df = n_samp - 1)

# stats.t.ppf(q = 0.975, 
#             df = n_samp - 1)

# sigma = sample.std()/np.sqrt(n_samp)

# stats.t.interval(0.95,
#                 df = n_samp - 1,
#                 loc = sample.mean(),
#                 scale= sigma)

# stats.t.interval(0.99,
#                 df = n_samp - 1,
#                 loc = sample.mean(),
#                 scale= sigma)


# #Comparing two Independent samples
# ttest_ind(sample1, 
#           sample2, 
#           equal_var = False)

# #comparing two paired samples (same sample)
# ttest_rel(before, 
#           after)

# #Anova test
# f_oneway(sample1, 
#          sample2, 
#          sample3)

#When doing a post test of the ANOVA you take the significant value 
#and divide it by the number of tests you did
#Bonferroni Correction

#reading in the data that has the driving distance of a golf ball after applying a new coating to it
#Current is old ball driving distance New is the driving distance after coating has been applied

# We are trying to see if the new coat on the ball 
# has made a significant difference in the driving distance

#reading in the data
df = pd.read_csv('Golf.csv')

#grabbing the columns
before = df['Current']
after = df['New']

threshold = 0.05

#performing the t-test to see if the new coating has made a significant difference
tscore, pval = ttest_ind(before, 
                         after, 
                         equal_var = False)

print("t-score: ", tscore)
print("p-value: ", pval)
print('Null hypothesis rejected') if pval < threshold else print('Null hypothesis accepted')
