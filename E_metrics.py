import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
from matplotlib.patches import Patch

from Z_utils import get_latest_file

'''Generates BoxPlot and agents response metrics.
Script generates 'response_data.xls' file in A_results folder
From that data it generates a boxplot and summary_table.xls in the same folder
The boxplot is generated directly from the 'response_data.xls'
The summary_table is just for information purpopses
'''

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")

def read_excel_data(file_path):
    xls = pd.ExcelFile(file_path)
    sheet_data = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        sheet_data[sheet_name] = df
    return sheet_data

def calculate_agent_metrics(agent_data):
    total_actions = len(agent_data)
    correct_actions = len(agent_data[agent_data['rew'] > 0])
    incorrect_actions = len(agent_data[agent_data['rew'] < 0])
    accuracy_ratio = (correct_actions / total_actions) * 100
    return accuracy_ratio, correct_actions, incorrect_actions

def calculate_metrics(data):
    accuracy_across_tabs = {}
    summary_data = []
    boxplot_data = []
    for sheet_name, df in data.items():
        for agent in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent]
            accuracy_ratio, correct_actions, incorrect_actions = calculate_agent_metrics(agent_data)
            if agent not in accuracy_across_tabs:
                accuracy_across_tabs[agent] = []
            accuracy_across_tabs[agent].append(accuracy_ratio)
            summary_data.append([agent, sheet_name, accuracy_ratio, correct_actions, incorrect_actions])
            boxplot_data.append([agent, accuracy_ratio])
    return summary_data, boxplot_data, accuracy_across_tabs

def generate_summary_table(accuracy_across_tabs):
    summary_list = []
    for agent, accuracies in accuracy_across_tabs.items():
        mean_accuracy = statistics.mean(accuracies)
        std_dev_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
        summary_list.append([agent, mean_accuracy, std_dev_accuracy])

    summary_table = pd.DataFrame(summary_list, columns=['Agent', 'Test Accuracy', 'Std Dev'])
    summary_table.to_excel(CURRENT_DIR+'/A_results/summary_table.xlsx', index=False)
    return summary_table


def generate_boxplot(boxplot_data):
    boxplot_df = pd.DataFrame(boxplot_data, columns=['Agent', 'Test Accuracy'])
    plt.figure(figsize=(10, 5))
    bp = boxplot_df.boxplot(column='Test Accuracy', by='Agent', patch_artist=True, return_type='dict', showmeans=True, meanline=True, meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.suptitle('')
    plt.xlabel('Agent')
    plt.ylabel('Accuracy (%)')

    # Styling the boxes in red
    for box in bp['Test Accuracy']['boxes']:
        box.set(color='red', linewidth=2)
        box.set(facecolor='red' )

    agent_names = sorted(boxplot_df['Agent'].unique())
    plt.xticks(range(1, len(agent_names) + 1), agent_names)
    plt.ylim(bottom=0)  # Set the y-axis limit starting from 0 %

    # Existing horizontal lines
    random_policy_line = plt.axhline(y=50, color='blue', linestyle='--', linewidth=2, label='Random Policy')
    r_5_line = plt.axhline(y=67, color='green', linestyle='--', linewidth=2, label='cluster r= 5%')

    # Adding the new horizontal line for "Tabular QL"
    tabular_ql_line = plt.axhline(y=20, color='purple', linestyle='--', linewidth=2, label='MARL QL')

    # Creating a legend and positioning it at the bottom right
    trained_policy_patch = Patch(color='red', label='DRL(ours)')
    plt.legend(handles=[random_policy_line, r_5_line, tabular_ql_line, trained_policy_patch], loc='lower right')

    plt.savefig(CURRENT_DIR+'/A_results/boxplot_' + TIMESTAMP+'.png', bbox_inches='tight')





# ====================
def main():
    '''This is the starting point of the script called by the Runner script'''
    file_path = get_latest_file(directory = os.path.join(CURRENT_DIR, 'A_results'), prefix = 'response_data', extension = '.xlsx') # Get the latest file in the directory by time stamp
    data      = read_excel_data(file_path)
    summary_data, boxplot_data, accuracy_across_tabs = calculate_metrics(data)
    summary_table = generate_summary_table(accuracy_across_tabs)
    generate_boxplot(boxplot_data)
    print(summary_table)


# ====================
# ====================
if __name__ == "__main__":
    main()
