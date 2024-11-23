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
stats.t.ppf(q = 0.025, 
            df = n_samp - 1)

stats.t.ppf(q = 0.975, 
            df = n_samp - 1)

sigma = sample.std()/np.sqrt(n_samp)

stats.t.interval(0.95,
                df = n_samp - 1,
                loc = sample.mean(),
                scale= sigma)

stats.t.interval(0.99,
                df = n_samp - 1,
                loc = sample.mean(),
                scale= sigma)


#Comparing two Independent samples
ttest_ind(sample1, 
          sample2, 
          equal_var = False)

#comparing two paired samples (same sample)
ttest_rel(before, 
          after)

#Anova test
f_oneway(sample1, 
         sample2, 
         sample3)

#When doing a post test of the ANOVA you take the significant value 
#and divide it by the number of tests you did
#Bonferroni Correction

