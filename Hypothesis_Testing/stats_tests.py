import scipy.stats as stats
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, f_oneway, chi2_contingency


# One-sample t-test (is the mean different from some value)
def one_sample_ttest(sample, population_mean, alpha=0.05, logger = None):
    t_stat, p_val = ttest_1samp(sample, population_mean)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    
    if logger is not None:
        logger.info(f"1-Sample t-test\n t={t_stat:.4f}, p={p_val:.4f} -> {result} H0")

    return t_stat, p_val, result

# Two-sample independent t-test
def independent_ttest(sample1, sample2, alpha=0.05, logger = None):
    t_stat, p_val = ttest_ind(sample1, sample2, equal_var=False)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    
    if logger is not None:
        logger.info(f"2-Sample Independent t-test\n t={t_stat:.4f}, p={p_val:.4f} -> {result} H0")

    return t_stat, p_val, result

# Paired t-test
def paired_ttest(before, after, alpha=0.05, logger = None):
    t_stat, p_val = ttest_rel(before, after)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    
    if logger is not None:
        logger.info(f"Paired t-test\n t={t_stat:.4f}, p={p_val:.4f} -> {result} H0")

    return t_stat, p_val, result

# One-way ANOVA
def anova_test(*groups, alpha=0.05, logger = None):
    f_stat, p_val = f_oneway(*groups)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    
    if logger is not None:
        logger.info(f"One-Way ANOVA\n F={f_stat:.4f}, p={p_val:.4f} -> {result} H0")

    return f_stat, p_val, result

# Chi-squared test of independence
def chi2_test(contingency_table, alpha=0.05, logger = None):
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    result = 'Reject' if p_val < alpha else 'Fail to Reject'
    
    if logger is not None:
        logger.info(f"Chi-squared Test\n χ²={chi2:.4f}, p={p_val:.4f} -> {result} H0")

    return chi2, p_val, result