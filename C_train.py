''' Trains a custom env
Works on Ray 2.6'''

import os, logging, sys, json, re, socket
import signal
import datetime
import gymnasium as gym
import numpy as np
import random
import pandas as pd
from openpyxl import Workbook
import ray
from ray import air, tune
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
from sigterm_handler import signal_handler, return_state_file_path # Resume after SIGTERM termination



#current_script_dir  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
#parent_dir          = os.path.dirname(current_script_dir)         # Get the parent directory (one level up)
#sys.path.insert(0, parent_dir)                                    # Add parent directory to sys.path

#from juwels.coalitions.B_env import DynamicCoalitionsEnv as Env   # custom environment
from B_env import DynamicCoalitionsEnv as Env                      # custom environment
from C_ppo_config import get_marl_trainer_config                   # Tranier config for PPO


output_dir = os.path.expanduser("~/ray_results") # Default output directory
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")



#*********************************** RUN RAY TRAINER *****************************************************
class RunRay:
    ''' Train a custom env on Ray 2.6'''

    def __init__(self, setup_dict,custom_env_config, experiment_name = 'shapley_rl'):

        current_dir      = os.path.dirname(os.path.realpath(__file__))
        self.excel_path  = os.path.join(current_dir, 'A_results', 'output.xlsx')
        self.jason_path  = os.path.join(current_dir, 'best_checkpoint_'+TIMESTAMP+'.json')
        self.clear_excel(self.excel_path)
        self.clear_json(self.jason_path)
        self.setup_dict = setup_dict
        self.custom_env_config = custom_env_config
        self.experiment_name   = experiment_name


    def setup_n_fit(self):
        '''Setup trainer dict and train model '''

        signal.signal(signal.SIGTERM, signal_handler) # Julich SIGTERM handling for graceful shutdown and restore unterminated runs
        experiment_path = os.path.join(output_dir, self.experiment_name)

        #_____________________________________________________________________________________________
        # Setup Config
        #_____________________________________________________________________________________________

        # TRAINER CONFIG - custom model (action and loss) and custom env
        train_batch_size_ = self.setup_dict['train_batch_size']
        seed              = self.setup_dict['seed']
        train_iteration   = self.setup_dict['training_iterations']
        NUM_CPUS          = self.setup_dict['cpu_nodes']
        lr_start,lr_end,lr_time = 2.5e-4,  2.5e-5, 50 * 1000000 #embelishments of the lr's

        # Get the trainer with the base configuration  - #OBS: no need to register Env anymore, as it is passed on the trainer config!
        trainer_config = get_marl_trainer_config(Env, self.custom_env_config,
                            train_batch_size_, lr_start, lr_time, lr_end, NUM_CPUS, seed,
                            fcnet_hiddens=[400, 400], entropy_coeff=0.03, num_sgd_iter=10,
                            kl_coeff=0.5, gamma=0.5, enable_learner_api=False)

        #_____________________________________________________________________________________________
        # Setup Trainer
        #_____________________________________________________________________________________________
        #callbacks       = WandbLoggerCallback(project=self.experiment_name, log_config=True, save_checkpoints = True) # local_mode = False only!

        # Conditional logic to decide whether to start a new experiment or restore an existing one
        state_file_path = return_state_file_path()
        if os.path.exists(state_file_path):
            with open(state_file_path, "r") as f:
                state = f.read().strip()
            if state == "interrupted":
                print("Previous run was interrupted. Attempting to restore...")
                tuner = tune.Tuner.restore(  #restore works only for unterminated runs
                path=experiment_path,
                trainable="PPO",
                resume_unfinished=True,
                resume_errored=False,
                restart_errored=False,
                param_space=trainer_config,  # Assuming `trainer_config` matches the original setup
            )
            os.remove(state_file_path) # Clear the state file after handling
        else: # Train from scratch
            print("Starting a new experiment run.")
            tuner  = tune.Tuner("PPO",
                    param_space = trainer_config,
                    run_config = air.RunConfig(
                        name =  self.experiment_name,
                        stop = {"training_iteration": train_iteration}, # "iteration" will be the metric used for reporting
                        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=100,
                                                           checkpoint_at_end=True,
                                                            num_to_keep= 5 ),#keep only the last 5 checkpoints
                        #callbacks = [wandb_callbacks],  # WandB local_mode = False only!
                        verbose= 2, #0 for less output while training - 3 for seeing custom_metrics better
                        local_dir = output_dir
                            )
                        )


        result_grid = tuner.fit() #train the model

         # Get reward per policy for the best iteration
        best_result_grid = result_grid.get_best_result(metric="episode_reward_mean", mode="max")

        # Print best training rewards per policy to console
        print("BEST ITERATION:")
        for key, value in best_result_grid.metrics["custom_metrics"].items():
            print(f"mean_{key}:", value)
        return  best_result_grid


    def train(self):
        ''' Calls Ray to train the model  '''
        #if ray.is_initialized(): ray.shutdown()
        #ray.init(local_mode=True,include_dashboard=False, ignore_reinit_error=True,log_to_driver=False, _temp_dir = '/p/home/jusers/cipolina-kun1/juwels/ray_tmp') # Default output directory') #local_mode=True it might run faster, used to debug. Set to false for WandB

        seeds_lst  = self.setup_dict['seeds_lst']
        for _seed in seeds_lst:
            self.set_seeds(_seed)
            print("we're on seed: ", _seed)
            self.setup_dict['seed'] = _seed
            best_res_grid      = self.setup_n_fit()
            result_dict        = self.save_results(best_res_grid,self.excel_path,self.jason_path, _seed) #print results, saves checkpoints and metrics

        ray.shutdown()
        return result_dict

    #____________________________________________________________________________________________
    #  Analize results and save files
    #____________________________________________________________________________________________

    def save_results(self, best_result_grid, excel_path, json_path, _seed):
        '''Save results to Excel file and save best checkpoint to JSON file
           :input: best_result_grid is supposed to bring the best iteration, but then we recover the entire history to plot
        '''

        # Process results
        df = best_result_grid.metrics_dataframe  #Access the entire *history* of reported metrics from a Result as a pd DataFrame. And not just the best iteration

        # Step 1: Identify columns for new average rewards per policy
        custom_reward_cols = [col for col in df.columns if col.startswith('custom_metrics/reward_policy')]

        # Identify columns for losses and entropies
        loss_cols = [col for col in df.columns if 'learner_stats/total_loss' in col]
        entropy_cols = [col for col in df.columns if 'learner_stats/entropy' in col]

        # Additional columns to include
        additional_cols = ["training_iteration", "episode_len_mean", "episode_reward_mean"]

        # Step 2: Create a new DataFrame with the desired columns
        desired_columns = custom_reward_cols + loss_cols + entropy_cols + additional_cols
        final_df = df[desired_columns]

        # Step 3: Save the new DataFrame to an Excel file
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a' if os.path.exists(excel_path) else 'w') as writer:
            final_df.to_excel(writer, sheet_name=f'Seed_{_seed}', index=False)

        # PRINT best iteration results to console
        print(df[additional_cols])

        # Save best checkpoint (i.e. the last) onto JSON filer
        best_checkpoints = []
        best_checkpoint = best_result_grid.checkpoint #returns a folder path, not a file.
        path_match      = re.search(r'Checkpoint\(local_path=(.+)\)', str(best_checkpoint))
        checkpoint_path = path_match.group(1) if path_match else None
        best_checkpoints.append({"seed": _seed, "best_checkpoint": checkpoint_path})
        with open(json_path, "a") as f:  # Save checkpoints to file
            json.dump(best_checkpoints, f, indent=4)

        return {'checkpoint_path': checkpoint_path, 'result_df': final_df}

    #____________________________________________________________________________________________
    # Aux functions
    #____________________________________________________________________________________________
    def set_seeds(self,seed):
        torch.manual_seed(seed)           # Sets seed for PyTorch RNG
        torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
        np.random.seed(seed=seed)         # Set seed for NumPy RNG
        random.seed(seed)                 # Set seed for Python's random RNG

    def clear_excel(self,excel_path):
        # Clear the existing Excel file
        if os.path.exists(excel_path):
            os.remove(excel_path)
        empty_wb = Workbook()     # Create a new empty Excel file
        empty_wb.save(excel_path)

    def clear_json(self,jason_path):
        with open(jason_path, "w") as f: pass #delete whatever was on the json file


#****************************************************************************************
# MAIN - run several seeds
#****************************************************************************************
if __name__=='__main__':

    '''TESTING CODE'''
    # SETUP for env and model
    setup_dict = {'training_iterations': 2,# 65,    # how many 'train_batches' will be collected'. Alternative: stop = {"episodes_total":1}, https://github.com/ray-project/ray/issues/8458
                  'train_batch_size'   : 400,   # how many steps
                  'seeds_lst'          : [42], #[ 300, 400,42] #100, 200,  #Train with different manually selected seeds (otherwise it just selects a seed for each worker)
                  }

    char_func_dict = {
        'mode'  : 'closed_form', #'ridesharing', #
        'alpha' : 1,
        'k'     : 1,
    }

    custom_env_config = {
        'num_agents'     : 4,
        'char_func_dict' : char_func_dict,
        'max_steps'      : 8000,
        'batch_size'     : 1000 # for the CV learning - one update per batch size
          }

    # Call Ray to train model
    train = RunRay(setup_dict,custom_env_config, experiment_name = 'shapley_rl').train()
