'''
Prints results of the last trained model and the best result
From here:
https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html
'''


from ray import tune, air
from ray.tune import ResultGrid
from ray.air import Result
'''
Brings back everything that was stored during training (basically, the results_grid))
'''

local_dir       = "/Users/lucia/ray_results"
exp_name        = 'turnEnv'  #NEEDS TO BE THE SAME as the one used on training.
experiment_path = f"{local_dir}/{exp_name}"
print(f"Loading results from {experiment_path}")

restored_tuner = tune.Tuner.restore(experiment_path)

##########################################################################################
#Latest trained model
##########################################################################################
result_grid = restored_tuner.get_results()
results_df = result_grid.get_dataframe()
#print(results_df[["training_iteration",'episode_reward_mean','episode_len_mean']])



##########################################################################################
#Best result
##########################################################################################

# Get the result with the maximum 'episode_reward_mean'
best_result: Result = result_grid.get_best_result(metric='episode_reward_mean', mode="max")

result_df = best_result.metrics_dataframe
print(result_df[["training_iteration",'episode_reward_mean','episode_len_mean']])

# Best result config dict (too verbose)
#print('config:',best_result.config)
