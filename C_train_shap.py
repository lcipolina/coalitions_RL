''' Trains a custom env
Works on Ray 2.6'''


import os, logging, sys, json, re, socket
import gymnasium as gym
import numpy as np
import random
import pandas as pd
from openpyxl import Workbook
import ray
from ray import air, tune
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec                        #For policy mapping
from typing import Dict                                               #for callbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks           # for callbacks
from ray.rllib.env import BaseEnv                                     # for callbacks
from ray.rllib.evaluation import RolloutWorker, Episode               # for callbacks
from ray.rllib.policy import Policy                                    #for callbacks
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.logger import UnifiedLogger


#current_script_dir  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
#parent_dir          = os.path.dirname(current_script_dir)         # Get the parent directory (one level up)
#sys.path.insert(0, parent_dir)                                    # Add parent directory to sys.path

from B_env_shap import ShapleyEnv as Env            # custom environment

# Define paths where to save results
home_dir       = '/Users/lucia/ray_results'
juelich_dir    = '/p/scratch/ccstdl/cipolina-kun/A-COALITIONS'
university_dir = '/home/zeta/Desktop/lucia/coalitions/coalitional_bargaining/ray_results'
# Get the hostname to determine the correct path
hostname = socket.gethostname().lower()
# Determine the output directory based on the hostname
if 'juwels' in hostname:
    output_dir = juelich_dir
elif 'zeta' in hostname:
    output_dir = university_dir
else:
    output_dir = os.path.expanduser("~/ray_results") # Default output directory

def custom_logger_creator( config = {"logdir": output_dir}):
    """Creates a custom logger with the specified path."""
    logdir = config.get("logdir", os.path.expanduser("~/ray_results")) # Define the directory where you want to store the logs
    os.makedirs(logdir, exist_ok=True)                                 # Ensure the directory exists
    return UnifiedLogger(config, logdir, loggers=None)                 # Return a UnifiedLogger object with the specified directory

#***************************************************************************************
######################################### TRAIN #########################################

#**********************************************************************
# Custom callbacks
# Get reward per agent (not provided in RLLIB)
# WandB Callbacks - Just logs results and metrics on each iteration
#***********************************************************************
class On_step_callback(DefaultCallbacks):
         '''To get rewards per agent
            Needs to be run with default 'verbose = 3' value to be displayed on screen
             #https://github.com/ray-project/ray/blob/master/rllib/evaluation/metrics.py#L229-L231
         '''
         def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: Episode,
                       **kwargs):
            '''Calculates the reward per agent at the STEP of the episode. Displays on Tensorboard and on the console   '''

            #The advantage of these ones is that it calculates the Max-Mean-Min and it prints on TB
            #NOTE: custom_metrics only take scalars
            my_dict = {}  #Needs this as metric name needs to be a string
            for key, values in episode.agent_rewards.items():
                my_dict[str(key)] = values

            episode.custom_metrics.update(my_dict)



#****************************************************************************************
class RunRay:
    ''' Train a custom env on Ray 2.6'''

    def __init__(self, setup_dict,custom_env_config, experiment_name = 'shapley_rl'):

        current_dir      = os.path.dirname(os.path.realpath(__file__))
        self.excel_path  = os.path.join(current_dir, 'A_results', 'output.xlsx')
        self.jason_path  = os.path.join(current_dir, 'best_checkpoint.json')
        self.clear_excel(self.excel_path)
        self.clear_json(self.jason_path)
        self.setup_dict = setup_dict
        self.custom_env_config = custom_env_config
        self.experiment_name   = experiment_name


    def setup(self):
        '''Setup trainer dict and train model '''

        # TRAINER CONFIG - custom model (action and loss) and custom env
        train_batch_size_ = self.setup_dict['train_batch_size']
        seed              = self.setup_dict['seed']
        train_iteration   = self.setup_dict['training_iterations']
        NUM_CPUS          = self.setup_dict['cpu_nodes']
        lr_start,lr_end,lr_time = 2.5e-4,  2.5e-5, 50 * 1000000 #embelishments of the lr's

        #_____________________________________________________________________________________________
        #DEFINE DIFFERENT POLICIES FOR EACH AGENT
        #1) Define the policies definition dict:
        # By using "PolicySpec(None...) empty, it will inferr the policy from the env
        env = Env(self.custom_env_config) #OBS: no need to register it anymore, as it is passed on the trainer config!
        def policy_dict():
            return {f"policy{i}": PolicySpec(observation_space=env.observation_space,
                            action_space=env.action_space,) for i in env._agent_ids}
        #2) Defines an agent->policy mapping function.
        # The mapping here is M (agents) -> N (policies), where M >= N.
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            '''Maps each Agent's ID with a different policy. So each agent is trained with a diff policy.'''
            get_policy_key = lambda agent_id: f"policy{agent_id}"
            return get_policy_key(agent_id)
        #_____________________________________________________________________________________________

        trainer_config    = (PPOConfig()
                .environment(env =Env, env_config= self.custom_env_config, disable_env_checking=True, ) #Register the Env!  (could't make the render work)
                .training(train_batch_size  = train_batch_size_,
                          model={"fcnet_hiddens": [400, 400]},
                          entropy_coeff     = 0.03,    #works well
                          num_sgd_iter      = 10,      #to make it faster
                          kl_coeff          = 0.5,      #kl loss, for policy stability
                          gamma             = 0.5,   #best results with DF = 1. If I decrease it, agents go bananas
                          lr                = lr_start,lr_schedule = [[0, lr_start],[lr_time, lr_end]], #good
                          _enable_learner_api=False #to keep using the old RLLIB API
                        )\
                .rollouts(num_rollout_workers=NUM_CPUS-1, num_envs_per_worker=1)
                .framework("torch")
                .rl_module(_enable_rl_module_api=False) #to keep using the old ModelCatalog API
                .multi_agent(  #EXTRA FOR THE POLICY MAPPING
                        policies = policy_dict(), #dict of policies
                        policy_mapping_fn = policy_mapping_fn #which pol in 'dict of pols' maps to which agent
                )\
                .callbacks(On_step_callback)\
                .debugging(seed=seed, logger_creator = custom_logger_creator )   #setting seed for reproducibility
            )

        #_____________________________________________________________________________________________
        # Setup Trainer
        #_____________________________________________________________________________________________
        #callbacks       = WandbLoggerCallback(project=self.experiment_name, log_config=True, save_checkpoints = True) # local_mode = False only!

        tuner  = tune.Tuner("PPO", param_space = trainer_config,
                                   run_config = air.RunConfig(
                                                name =  self.experiment_name,
                                                stop = {"training_iteration": train_iteration}, # metric used for reporting
                                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=100, checkpoint_at_end=True,
                                                                                       num_to_keep= 5 #keep only the last 5
                                                                                       ),
                                                #callbacks = [callbacks],  # WandB local_mode = False only!
                                                verbose= 1, #less output while training - 3 for seeing custom_metrics better
                                                local_dir = output_dir
                                            )
                                    )

        result_grid = tuner.fit()

        # Get reward per policy
        best_result_grid = result_grid.get_best_result(metric="episode_reward_mean", mode="max")

        # Display training results
        for key, value in best_result_grid.metrics["policy_reward_mean"].items():     # Print rewards per policy to console
            print(f"reward_mean_{key}:", value)

        return  best_result_grid


    def train(self):
        ''' Calls Ray to train model  '''
        if ray.is_initialized(): ray.shutdown()
        ray.init(local_mode=True,include_dashboard=False, ignore_reinit_error=True,log_to_driver=False) #local_mode=True it might run faster, used to debug. Set to false for WandB

        res_list   = ["training_iteration",'episode_len_mean','episode_reward_mean' ] #columns saved into excel
        seeds_lst  = self.setup_dict['seeds_lst']

        for _seed in seeds_lst:
            self.set_seeds(_seed)
            print("we're on seed: ", _seed)
            self.setup_dict['seed'] = _seed
            best_res_grid      = self.setup()
            result_dict        = self.save_results(best_res_grid,res_list,self.excel_path,self.jason_path, _seed) #print results, saves checkpoints and metrics

        ray.shutdown()
        return result_dict

    #____________________________________________________________________________________________
    #  Analize results and save files
    #____________________________________________________________________________________________
    def save_results(self,best_result, res_list,excel_path, json_path, _seed):

        # PROCESS RESULTS
        # Print Best Results to console
        result_df  = best_result.metrics_dataframe #Brings the results saved on the 'progress.csv' file as dataframe
        print("BEST RESULTS:")
        print(result_df[res_list]) #dataframe slicing

        # SAVE BEST CHECKPOINT (i.e. the last) onto JSON filer
        best_checkpoints = []
        best_checkpoint = best_result.checkpoint #returns a folder path, not a file.
        path_match      = re.search(r'Checkpoint\(local_path=(.+)\)', str(best_checkpoint))
        checkpoint_path = path_match.group(1) if path_match else None
        best_checkpoints.append({"seed": _seed, "best_checkpoint": checkpoint_path})

        # Save checkpoints
        with open(json_path, "a") as f:
            json.dump(best_checkpoints, f, indent=4)

        # Save results onto Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl',mode='a') as writer:
                result_df[res_list].to_excel(writer, sheet_name=str(_seed), index=False)

        return {'checkpoint_path': checkpoint_path, 'result_df': result_df}

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
