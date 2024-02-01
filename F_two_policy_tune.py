''' Hyperparam tuning on PPO
'''

import os
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, Episode
from typing import Dict
from ray.rllib.policy import Policy

#Import environment definition
from A_two_policy_env import TurnEnv as Env


###############################################################
#     Tune & Train with RLLIB
###############################################################

#**********************************************************************
# Custom callbacks
# Get reward per agent (not provided in RLLIB)
#***********************************************************************

class MyCallback(DefaultCallbacks):
         '''To get rewards per agent
            Needs to be run with default 'verbose' value
         '''
         def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: Episode,
                       **kwargs):
            '''Calculates the reward per agent at the STEP of the episode. Displays on Tensorboard  and on the console   '''

            #The advantage of these ones is that it calculates the Max-Mean-Min and it prints on TB
            #NOTE: custom_metrics only take scalars
            my_dict = {}  #Needs this as metric name needs to be a string
            for key, values in episode.agent_rewards.items():
                my_dict[str(key)] = values
            episode.custom_metrics.update(my_dict)


            # This will give the dict of all agents and values printed on each episode - worse than my own printout
            #print('episode.agent_rewards',list(episode.agent_rewards.values()))

            # Graphs of metrics over time. #TODO -   # The hist_data attribute will create for you histogram and distributions in TensorBoard whereas the custom_metrics create scalars.
            #episode.custom_metrics["return_hist"] = episode.hist_data["mean_return_per_agent"]


#**********************************************************************
# Driver code for training
#**********************************************************************
def train_ray():
    ''' Registers Env and sets the dict for PPO
    '''

    #**********************************************************
    # Define configuration with hyperparam and training details
    #**********************************************************
    env_name = 'TurnEnv'
    tune.register_env(env_name, lambda env_ctx: Env()) #the register_env needs a callable/iterable


    #Experiment configuration
    NUM_CPUS = os.cpu_count()

    #hyperparam tuning
    config = PPOConfig()\
    .framework("torch")\
    .rollouts(num_rollout_workers=1, observation_filter="MeanStdFilter")\
    .resources(num_gpus=0,num_cpus_per_worker=1)\
    .evaluation(evaluation_interval=2,evaluation_duration = 2, evaluation_duration_unit='episodes',
                evaluation_config= {"explore": False})\
    .environment(env = env_name, env_config={
                                     "num_workers": NUM_CPUS - 1,
                                     "disable_env_checking":True} #env_config: arguments passed to the Env + num_workers = # number of parallel workers
                )\
    .training(lr                 = tune.grid_search([1e-2, 1e-3, 1e-4]), #PPO-specific params
              num_sgd_iter       = tune.grid_search([10, 20, 30]),  #epoch range
              train_batch_size   = tune.grid_search([600,800,1000]),
              sgd_minibatch_size = tune.grid_search([128,256,512]),
              #gamma              = tune.grid_search([ 0.9, 0.99]),   #df
              grad_clip          = 0.95,
              vf_clip_param      = 10.0,
              #clip_param         = tune.grid_search([ 0.1, 0.2, 0.3]),
              #entropy_coeff      = 0.01, #grid_search on this param gives an error
              lambda_            = tune.grid_search([ 0.9, 0.95, 0.99]), #reward shaping og GAE
              shuffle_sequences  = True
               )

    #RLLIB callbacks
    config.callbacks(MyCallback)


    if ray.is_initialized(): ray.shutdown()
    ray.init(local_mode=True,include_dashboard=False, ignore_reinit_error=True) #If dashboard True, prints the dashboard running on a local port
    #ray.init(local_mode=True) # it might run faster, used to debug. Use: num_workers=0

    train_steps = 1
    experiment_name = 'TurnEnv'
    stop_timesteps = 1
    stop_iters = 1
    stop_dict = {
        "timesteps_total": stop_timesteps,
        "training_iteration": stop_iters,
    }

    tuner = tune.Tuner("PPO", param_space=config,
                              run_config=air.RunConfig(
                                        name =  experiment_name,
                                        stop=stop_dict,
                                        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50, checkpoint_at_end=True)
                                        #verbose= 0
                                )
                     )
    results = tuner.fit()

    #check_learning_achieved(results, stop_reward)
    df = results.get_dataframe()
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    #print(best_result)


    # PRINT BEST SET OF HYPERPARAMS ##########################################################################
    hyperparam_mutations = ['lr','num_sgd_iter','train_batch_size','sgd_minibatch_size','gamma','grad_clip','lambda',
                            'vf_clip_param', 'clip_param','entropy_coeff','shuffle_sequences']

    import pprint
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint(
        {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    )
    print("\nBest performing trial's final reported metrics:\n")
    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

    ray.shutdown()

###############################################################
#     Test the Env step by step (without RLLIB)
###############################################################
def test_env():
    env = Env()

    for _ in range(2):
        obs = env.reset()
        env.step(obs)

        print('reset_obs:', obs)
        print('env.action_space: ',env.action_space)
        #actions =env.action_space[0].sample(),env.action_space[1].sample()
        obs, reward, dones, info = env.step(env.action_space) #Note: every return is a dict
        print('OBS:',obs.values())
        #print('REWARD:', reward)




if __name__ == '__main__':

   ##################################
   # Run only one:
   ##################################

   train_ray()

   #test_env()


'''
Best performing trial's final set of hyperparameters:
{'clip_param': 0.3,
 'entropy_coeff': 0.0,
 'gamma': 0.99,
 'grad_clip': 0.95,
 'lr': 0.01,
 'num_sgd_iter': 10,
 'sgd_minibatch_size': 128,
 'shuffle_sequences': True,
 'train_batch_size': 1000,
 'vf_clip_param': 10.0}


'''