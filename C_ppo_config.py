''' Defines the Config Dict for PPO used for training and inference'''


from ray.rllib.policy.policy import PolicySpec                        # for policy mapping
from typing import Dict                                               # for callbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks           # for callbacks
from ray.rllib.env import BaseEnv                                     # for callbacks
from ray.rllib.evaluation import RolloutWorker, Episode               # for callbacks
from ray.rllib.policy import Policy                                   # for callbacks
from ray.air.integrations.wandb import WandbLoggerCallback
#from ray.tune.logger import UnifiedLogger

from ray.rllib.algorithms.ppo import PPOConfig


#**********************************************************************
# Custom callbacks
# Get reward per agent (not provided in RLLIB)
# WandB Callbacks - Just logs results and metrics on each iteration
#***********************************************************************
class Custom_callback(DefaultCallbacks):
    '''To get rewards per agent - data is stored in episode.custom_metrics
            Needs to be run with default 'verbose = 3' value to be displayed on screen
             #https://github.com/ray-project/ray/blob/master/rllib/evaluation/metrics.py#L229-L231
             episode.agent_rewards in RLLIB  contains the accumulated rewards for each agent per episode.
    '''
    '''Taylored for turn-based environments where each agent acts once per episode.'''


    def on_train_result(self, *, algorithm, result: dict, **kwargs):
         # Method called at the end of each training iteration to average rewards per iteration
        episodes_this_iter = result.get("episodes_this_iter", 0)

        if episodes_this_iter > 0:
            num_agents = algorithm.workers.local_worker().env_context['num_agents']

            for agent_id in range(num_agents):
                key = f"reward_policy{agent_id}"
                if key in result["custom_metrics"]:
                    # Focus on the last 'episodes_this_iter' entries for the current iteration - given that RLLIB accumulates rewards across episodes
                    recent_rewards = result["custom_metrics"][key][-episodes_this_iter:]
                    average_reward = sum(recent_rewards) / episodes_this_iter
                    result["custom_metrics"][key] = average_reward

        #print("Updated custom metrics with averages:", result["custom_metrics"])

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                   policies: Dict[str, Policy], episode: Episode,
                   **kwargs):
        # Accumulate rewards per episode (i.e. sums rewwards across steps until the reset calls  - i.e. episode ends)
        env = base_env.get_sub_environments()[0]  # Use get_sub_environments() instead of get_unwrapped()
        accumulated_rewards = env.accumulated_rewards

        for agent_id, reward in accumulated_rewards.items():
            key = f"reward_policy{agent_id}"
            if key in episode.custom_metrics:
                episode.custom_metrics[key] += reward
            else:
                episode.custom_metrics[key] = reward
       # print("Accumulated rewards:", accumulated_rewards)


        ''' If it weren't a turn-based env, we could use this one:
        def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: Episode,
                       **kwargs):
            #Calculates and shows the SUM of the reward dict per agent at each STEP of the episode. Displays on Tensorboard and on the console
            #The advantage of these ones is that it calculates the Max-Mean-Min and it prints on TB
            #NOTE: custom_metrics only take scalars
            for key, values in episode.agent_rewards.items():
                episode.custom_metrics[f"reward_{key}"] = values
                print(f"reward_{key}:", values)
        '''


#**********************************************************************
# DEFINE DIFFERENT POLICIES FOR EACH AGENT
# 1) Define the policies definition dict:
# By using "PolicySpec(None...) empty, it will inferr the policy from the env
#***********************************************************************

def policy_dict(env):
    return {f"policy{i}": PolicySpec(observation_space=env.observation_space,
                            action_space=env.action_space,) for i in env._agent_ids}
# 2) Defines an agent->policy mapping function.
# The mapping here is M (agents) -> N (policies), where M >= N.
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    '''Maps each Agent's ID with a different policy. So each agent is trained with a diff policy.'''
    get_policy_key = lambda agent_id: f"policy{agent_id}"
    return get_policy_key(agent_id)



#**********************************************************************
# Create PPO config dict
# Including the CustomCallback
#***********************************************************************



def get_marl_trainer_config(env_generator, custom_env_config,setup_dict, lr_start=None, lr_time=None, lr_end=None,
                            fcnet_hiddens=[400, 400], entropy_coeff=0.03, num_sgd_iter=10,
                            kl_coeff=0.5, gamma=0.5, enable_learner_api=False):
    """
    Generates a base training configuration for PPO with customizable options.

    Parameters:
    - env: The environment class or string identifier. NOT the instance of a registered env.
    Returns:
    - An instance of PPOConfig with the specified configuration.
    """

    train_batch_size_ = setup_dict['train_batch_size']
    seed              = setup_dict['seed']
    NUM_CPUS          = setup_dict['cpu_nodes']

    env_object = env_generator(custom_env_config)
    config_dict = (PPOConfig()
            .environment(env=env_generator, env_config=custom_env_config, disable_env_checking=True)  # Register the Env!
            .training(train_batch_size=train_batch_size_,
                      model={"fcnet_hiddens": fcnet_hiddens},
                      entropy_coeff=entropy_coeff,  # works well
                      num_sgd_iter=num_sgd_iter,  # to make it faster
                      kl_coeff=kl_coeff,  # kl loss, for policy stability
                      gamma=gamma,  # best results with DF = 1. If I decrease it, agents go bananas
                      lr=lr_start,
                      lr_schedule=[[0, lr_start], [lr_time, lr_end]],  # good
                      _enable_learner_api=enable_learner_api)  # to keep using the old RLLib API
            .rollouts(num_rollout_workers=NUM_CPUS-1, num_envs_per_worker=1, rollout_fragment_length='auto')
            .framework("torch")
            .rl_module(_enable_rl_module_api=False)  # to keep using the old ModelCatalog API
            .multi_agent(policies=policy_dict(env_object), policy_mapping_fn=policy_mapping_fn)  # EXTRA FOR THE POLICY MAPPING
            .callbacks(Custom_callback)
            .debugging(seed=seed)  # setting seed for reproducibility
            .reporting(keep_per_episode_custom_metrics=True)
            )

    #.debugging(seed=seed, logger_creator = custom_logger_creator )
    return config_dict