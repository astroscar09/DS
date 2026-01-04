from plotting_utils import *
from stats_tests import *
from data import *
from scipy.stats import shapiro


# Example usage
if __name__ == "__main__":
    import yaml 

    yaml_path = 'config.yaml'

    with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

    data_file = config['file']
    plot_dir = config['plot_dir']

    # Load data
    print("=== Loading Data ===")
    df = load_data(data_file)
    before = df['Current']
    after = df['New']
    print(f"Data loaded: {len(before)} samples before, {len(after)} samples after")

    print(f'Testing the normality of the data...')
    differences = after - before

    hist_plot_differences(differences, save_file=f'{plot_dir}/distance_difference.pdf')
    probplot_differences(differences, save_file=f'{plot_dir}/Quantile_Plots_Against_Normal_Distribution.pdf')

    stat, p = shapiro(differences)

    print(f"Shapiro-Wilk test: W={stat:.4f}, p-value={p:.4f} \n")

    if p > 0.05:
        print("The differences appear normally distributed.")
    else:
        print("The differences do not appear normally distributed.")


    print("\n=== Exploratory Plots ===")
    plot_histograms(before, after)

    print("\n=== Paired t-test ===")
    paired_ttest(before, after)

    print("\n=== Independent t-test ===")
    independent_ttest(before, after)

    print("\n=== One-sample t-test (e.g. comparing to population mean 250) ===")
    one_sample_ttest(after, 250)
