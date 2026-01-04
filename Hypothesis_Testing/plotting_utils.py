import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.family'] = 'Sans-serif'


# Plot distribution
def plot_histograms(before, after, labels=("Before", "After"), save_file = None):
    sns.histplot(before, kde=True, label=labels[0], color="blue")
    sns.histplot(after, kde=True, label=labels[1], color="orange")
    plt.legend()
    plt.title("Distribution Comparison")
    #plt.show()
    if save_file is not None:
        plt.savefig(save_file)

def hist_plot_differences(differences, save_file = None):

    plt.figure(figsize=(12, 6))
    sns.histplot(differences, kde=True)
    plt.title('Distribution of Differences (After - Before)')
    if save_file is not None:
        plt.savefig(save_file)

def probplot_differences(differences, save_file = None):

    plt.figure(figsize=(12, 6))
    stats.probplot(differences, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Differences")
    if save_file is not None:
        plt.savefig(save_file)
