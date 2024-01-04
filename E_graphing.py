'''the calculation of the mean and confidence interval is done implicitly by the Seaborn lineplot function.
For each lineplot call, Seaborn takes the input data and calculates the mean and confidence interval automatically. The ci="sd" parameter tells Seaborn to use the standard deviation as the measure of uncertainty (confidence interval) for the mean.
The mean is calculated for each unique value of the x-axis (in this case, training_iteration) and is plotted as the line. The confidence interval is displayed as a shaded area around the mean line.
'''


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import load_workbook

current_dir = os.path.dirname(os.path.realpath(__file__))
excel_path = os.path.join(current_dir, 'A_results', 'output.xlsx')

# Delete useless "Sheet1"
book = load_workbook(excel_path)
sheets = book.sheetnames
if 'Sheet' in sheets:
    # If yes, remove "Sheet1"
    std = book['Sheet']
    book.remove(std)
book.save(excel_path)

# Read all sheets from the Excel file
sheets = pd.read_excel(excel_path, sheet_name=None)

# Create empty dataframes for each policy group
pol0_df = pd.DataFrame()
pol1_df = pd.DataFrame()

for sheet_name, df in sheets.items():
    df = df.set_index('training_iteration')

    # Filter columns by the presence of '0' or '1' in their names
    pol0_columns = [col for col in df.columns if '0' in col and col != 'episode_len_mean']
    pol1_columns = [col for col in df.columns if '1' in col and col != 'episode_reward_mean']

    pol0_temp_df = df[pol0_columns].melt(ignore_index=False, var_name='Policy', value_name='Reward').reset_index()
    pol1_temp_df = df[pol1_columns].melt(ignore_index=False, var_name='Policy', value_name='Reward').reset_index()

    pol0_df = pol0_df.append(pol0_temp_df)
    pol1_df = pol1_df.append(pol1_temp_df)

sns.set_style("whitegrid")
fig, ax = plt.subplots()

# Plot using seaborn lineplot
sns.lineplot(x="training_iteration", y="Reward", data=pol0_df, ci="sd", label="Policy 0", ax=ax)
sns.lineplot(x="training_iteration", y="Reward", data=pol1_df, ci="sd", label="Policy 1", ax=ax)

ax.set_xlabel("Training Iteration")
ax.set_ylabel("Average Rewards")
ax.set_title("Rewards by Policy")

# Add this line to adjust y-axis ticks
ax.yaxis.set_ticks(np.arange(0, 14, 2))

plt.legend()
fig.savefig("mean_and_confidence_interval_plot.pdf", bbox_inches='tight')

plt.show()


