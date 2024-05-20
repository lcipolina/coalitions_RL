'''Script to reproduce a boxplot from a table in an Excel file'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np

# Load the Excel file
file_path = '/Users/lucia/Desktop/LuciaArchive/000_A_MY_RESEARCH/00-My_Papers/Ridesharing/000-A-RidesharingMARL/00-Codes/coalitions/A-coalitions_paper/A_results/summary_table_neurips.xlsx'
#xls = pd.ExcelFile(file_path)

df = pd.read_excel(file_path, sheet_name='Sheet1')



# Simulate data points based on mean and standard deviation
np.random.seed(0)  # For reproducibility
simulated_data = []

for _, row in df.iterrows():
    agent = row['Agent']
    mean = row['Test Accuracy']
    std_dev = row['Std Dev']
    # Simulate 1000 data points per agent
    simulated_accuracies = np.random.normal(loc=mean, scale=std_dev, size=1000)
    simulated_data.append(pd.DataFrame({'Agent': agent, 'Test Accuracy': simulated_accuracies}))

# Combine all simulated data into a single DataFrame
simulated_df = pd.concat(simulated_data)

# Define the function to generate the boxplot
def generate_boxplot(data):
    plt.figure(figsize=(12, 6))

    # Create the boxplot
    boxplot = data.boxplot(column='Test Accuracy', by='Agent', patch_artist=True, showmeans=True, meanline=True,
                           meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'})

    plt.suptitle('')
    plt.xlabel('Agent')
    plt.ylabel('Accuracy (%)')
    plt.title('Boxplot of Test Accuracy by Agent')

    # Styling the boxes in red
    for box in boxplot.artists:
        box.set_edgecolor('red')
        box.set_linewidth(2)
        box.set_facecolor('red')

    # Set agent names and y-axis limit
    agent_names = sorted(data['Agent'].unique())
    plt.xticks(range(1, len(agent_names) + 1), agent_names)
    plt.ylim(bottom=0)  # Set the y-axis limit starting from 0 %

    # Adding horizontal lines
    random_policy_line = plt.axhline(y=50, color='blue', linestyle='--', linewidth=2, label='Random Policy')
    r_5_line = plt.axhline(y=67, color='green', linestyle='--', linewidth=2, label='cluster = 5%')
    tabular_ql_line = plt.axhline(y=20, color='purple', linestyle='--', linewidth=2, label='MARL QL')

    # Creating a legend and positioning it at the bottom right
    trained_policy_patch = Patch(color='red', label='DRL(ours)')
    plt.legend(handles=[random_policy_line, r_5_line, tabular_ql_line, trained_policy_patch], loc='lower right')

    # Save the plot
    plt.savefig('boxplot_from_table.png')
    plt.show()

# Generate the boxplot using the simulated data
generate_boxplot(simulated_df)