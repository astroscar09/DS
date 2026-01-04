import scipy.stats as stats
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, f_oneway, chi2_contingency


# One-sample t-test (is the mean different from some value)
def one_sample_ttest(sample, population_mean, alpha=0.05):
    t_stat, p_val = ttest_1samp(sample, population_mean)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    print(f"1-Sample t-test\n t={t_stat:.4f}, p={p_val:.4f} -> {result} H0")

# Two-sample independent t-test
def independent_ttest(sample1, sample2, alpha=0.05):
    t_stat, p_val = ttest_ind(sample1, sample2, equal_var=False)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    print(f"2-Sample Independent t-test\n t={t_stat:.4f}, p={p_val:.4f} -> {result} H0")

# Paired t-test
def paired_ttest(before, after, alpha=0.05):
    t_stat, p_val = ttest_rel(before, after)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    print(f"Paired t-test\n t={t_stat:.4f}, p={p_val:.4f} -> {result} H0")

# One-way ANOVA
def anova_test(*groups, alpha=0.05):
    f_stat, p_val = f_oneway(*groups)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    print(f"One-Way ANOVA\n F={f_stat:.4f}, p={p_val:.4f} -> {result} H0")

# Chi-squared test of independence
def chi2_test(contingency_table, alpha=0.05):
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    print(f"Chi-squared Test\n χ²={chi2:.4f}, p={p_val:.4f} -> {result} H0")