from plotting_utils import *
from stats_tests import *
from data import *
from scipy.stats import shapiro
from logging_utils import *

# Example usage
if __name__ == "__main__":
    import yaml 

    yaml_path = 'config.yaml'

    with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

    data_file = config['file']
    plot_dir = config['plot_dir']
    log_dir = config['log_dir']

    logger = setup_logger(log_dir="logs", log_name='AB_Testing_Golf_Ball.log')

    # Load data
    logger.info("=== Loading Data ===")
    df = load_data(data_file)
    before = df['Current']
    after = df['New']
    logger.info(f"Data loaded: {len(before)} samples before, {len(after)} samples after")

    logger.info(f'Testing the normality of the data...')
    differences = after - before

    #print("\n=== Exploratory Plots ===")
    plot_histograms(before, after, save_file=f'{plot_dir}/checking_normality_in_samples.pdf')
    hist_plot_differences(differences, save_file=f'{plot_dir}/distance_difference.pdf')
    probplot_differences(differences, save_file=f'{plot_dir}/Quantile_Plots_Against_Normal_Distribution.pdf')

    stat, p = shapiro(differences)

    logger.info(f"Shapiro-Wilk test: W={stat:.4f}, p-value={p:.4f} \n")

    if p > 0.05:
        logger.info("The differences appear normally distributed.")
    else:
        logger.info("The differences do not appear normally distributed.")

    logger.info("\n=== Paired t-test ===")
    paired_t_stat, paired_t_p_val, paired_t_result = paired_ttest(before, after, logger = logger)

    logger.info("\n=== Independent t-test ===")
    ind_t_stat, ind_t_stat_p_val, ind_t_stat_result  = independent_ttest(before, after, logger = logger)

    logger.info("\n=== One-sample t-test (e.g. comparing to population mean of old golf ball used) ===")
    current_ball_mean_dist = df.mean()['Current']


    one_sample_t_stat, one_sample_t_stat_p_val, one_sample_t_stat_result  = one_sample_ttest(after, current_ball_mean_dist, logger = logger)

    results_dict = {'Test': ['Paired-t-test', 'Independent-t-test', 'one-sample_t-test'], 
                    't-stats': [paired_t_stat, ind_t_stat, one_sample_t_stat], 
                    'p_vals': [paired_t_p_val, ind_t_stat_p_val, one_sample_t_stat_p_val], 
                    'Reject Null': [paired_t_result, ind_t_stat_result, one_sample_t_stat_result]}

    results_path = 'results/ttest_results.csv'

    logger.info(f'Results saved at: {results_path}')
    pd.DataFrame(results_dict).to_csv(results_path)
