''' Mean Reward, Entropy and Loss from RLLIB's output

The calculation of the mean and confidence interval is done implicitly by the Seaborn lineplot function.
For each lineplot call, Seaborn takes the input data and calculates the mean and confidence interval automatically. The ci="sd" parameter tells Seaborn to use the standard deviation as the measure of uncertainty (confidence interval) for the mean.
The mean is calculated for each unique value of the x-axis (in this case, training_iteration) and is plotted as the line. The confidence interval is displayed as a shaded area around the mean line.
'''

import datetime
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import load_workbook

current_dir = os.path.dirname(os.path.realpath(__file__))
excel_path = os.path.join(current_dir, 'A_results', 'output.xlsx')
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")

def graph_reward_n_others(rew_good_action=10, rew_bad_action=-100):
    '''Plot mean and variance across several seeds'''

    current_dir = os.path.dirname(os.path.realpath(__file__))
    excel_path = os.path.join(current_dir, 'A_results', 'output.xlsx')
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Delete useless "Sheet1"
    book = load_workbook(excel_path)
    sheets = book.sheetnames
    if 'Sheet' in sheets:
        std = book['Sheet']
        book.remove(std)
    book.save(excel_path)

    # Read all sheets from the Excel file
    sheets = pd.read_excel(excel_path, sheet_name=None)

    # Create dictionaries to hold dataframes for each metric
    policy_reward_dfs, policy_loss_dfs, policy_entropy_dfs = {}, {}, {}
    unique_policies = set()

    for sheet_name, df in sheets.items():
        df = df.set_index('training_iteration')

        # Process new reward policy columns and existing metrics
        for col in df.columns:
            if col.startswith('custom_metrics/reward_policy'):
                # New reward metrics
                policy_name = col.split('/')[-1]
                temp_df = df[[col]].rename(columns={col: 'Reward'}).reset_index()
                temp_df['Policy'] = policy_name
                policy_reward_dfs[policy_name] = policy_reward_dfs.get(policy_name, pd.DataFrame())
                policy_reward_dfs[policy_name] = pd.concat([policy_reward_dfs[policy_name], temp_df])
            elif 'learner_stats/total_loss' in col or 'learner_stats/entropy' in col:
                # Existing metrics (loss and entropy)
                policy_name = col.split('/')[2]
                metric_name = 'Loss' if 'total_loss' in col else 'Entropy'
                temp_df = df[[col]].rename(columns={col: metric_name}).reset_index()
                temp_df['Policy'] = policy_name
                target_dict = policy_loss_dfs if metric_name == 'Loss' else policy_entropy_dfs
                target_dict[policy_name] = target_dict.get(policy_name, pd.DataFrame())
                target_dict[policy_name] = pd.concat([target_dict[policy_name], temp_df])

        # Update unique_policies set
        unique_policies.update(policy_reward_dfs.keys())

        # Update unique_policies set
        unique_policies.update(policy_reward_dfs.keys())

    # Calculate Max Theoretical Reward and Random Policy AVG Reward per agent
    number_of_policies = len(unique_policies)
    max_theoretical_reward = (2 ** (number_of_policies - 1)) * rew_good_action
    rnd_pol_rew = (2 ** (number_of_policies - 1)) *(0.5 * rew_good_action + 0.5 * rew_bad_action)

    # Plotting and Saving the Plots
    sns.set_style("whitegrid")
    metrics_data = {
        'Reward': policy_reward_dfs,
        'Loss': policy_loss_dfs,
        'Entropy': policy_entropy_dfs
    }
    for metric, dfs in metrics_data.items():
        fig, ax = plt.subplots()
        for policy, data in dfs.items():
            sns.lineplot(x="training_iteration", y=metric, data=data, ci='sd', label=policy, ax=ax)

        # Add Max Theoretical Reward and Random Policy AVG Reward lines for Reward metric
        if metric == 'Reward':
            # Add Max Theoretical Reward line
            max_reward_data = pd.DataFrame({
                'training_iteration': dfs[next(iter(dfs))]['training_iteration'],
                'Max Theoretical Return': [max_theoretical_reward] * len(dfs[next(iter(dfs))])
            })
            sns.lineplot(x='training_iteration', y='Max Theoretical Return', data=max_reward_data, ax=ax, label='Max Theoretical Return', linestyle='--', linewidth=2.5, color='red')
            # Add Random Policy AVG Reward line
            rnd_reward_data = pd.DataFrame({
                'training_iteration': dfs[next(iter(dfs))]['training_iteration'],
                'Random Policy AVG Return': [rnd_pol_rew] * len(dfs[next(iter(dfs))])
            })
            sns.lineplot(x='training_iteration', y='Random Policy AVG Return', data=rnd_reward_data, ax=ax, label='Random Policy Return', linestyle='--', linewidth=2.5, color='hotpink')

        if metric =='Reward':
            metric = 'Returns' # What RLLIB calls 'Rewards' are actually "Undiscounted Returns"
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel(f"{metric}")
        ax.set_title(f" Training {metric} by Policy")
        plt.legend()
        fig.savefig(os.path.join(current_dir, f"A_results/{metric.lower()}_plot_{TIMESTAMP}.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    graph_reward_n_others()
    print('DONE!')
