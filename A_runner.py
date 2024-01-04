from B_env_shap import ShapleyEnv as Env
from C_train_shap import RunRay as train_policy
from D_deploy import Inference as evaluate_policy
from E_metrics import main as metrics
from G_distance_lst import DistanceGenerator



class ShapleyRunner:
    def __init__(self, setup_dict, char_func_dict, distance_gen_config):
        # Generate list of list of rnd distances and their testing counterparts - they should be generated at the same time to be chained.
        # revisit_steps should be less or equal num_steps

        self.distance_gen = DistanceGenerator(
            grid_interval =distance_gen_config['grid_interval'],
            num_agents    =distance_gen_config['num_agents'],
            num_steps     =distance_gen_config['num_steps_training'],
            revisit_steps =distance_gen_config['revisit_steps'],
            epsilon       =distance_gen_config['epsilon']
        )
        self.train_distance_lst = self.distance_gen.training_distances
        # less observations as we only update every iteration. Num_obs = num_iterations -1
        self.test_distance_lst = self.distance_gen.generate_testing_distances(num_obs=distance_gen_config['num_steps_testing']) #same as train but moved by an epsilon

        # Initialize other configs
        self.setup_dict     = setup_dict
        self.char_func_dict = char_func_dict
        # Pass the distance list to the env for CV learning
        self.custom_env_config = {
            'num_agents'      : distance_gen_config['num_agents'],
            'char_func_dict'  : char_func_dict,
            'manual_distances': self.train_distance_lst,
            'max_steps'       : 10000, #not used
            'batch_size'      : setup_dict['train_batch_size']
        }

    def train(self):
        train_result         = train_policy(self.setup_dict, self.custom_env_config, self.setup_dict['experiment_name']).train()
        self.checkpoint_path = train_result['checkpoint_path']
        return self.checkpoint_path

    def evaluate(self, checkpoint_path=None, test_distance_lst=None):

        if test_distance_lst is None:
           test_distance_lst=self.test_distance_lst

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        eval_cls = evaluate_policy(checkpoint_path, self.custom_env_config)
        responses_by_distance, final_coalitions_by_distance = eval_cls.play_env_custom_obs(distance_lst_input=test_distance_lst)
        eval_cls.excel_n_plot(responses_by_distance, final_coalitions_by_distance)
        metrics() # generates boxplot and other graphs




##################################################
# CONFIGS
##################################################
if __name__ == '__main__':

    setup_dict = {
        'training_iterations': 3, #(10*24),      # we need ~20 per each distance to learn 100%. Iterations = num_distances * 20
        'train_batch_size'   : 500, #2800,        # we need approx 2200 steps to learn 100%
        'seeds_lst'          : [100],
        'experiment_name'    :'shapley_rl',
        'cpu_nodes'         : 3 # 3 for home, change for Zeta or 40 for Juelich.
    }

    char_func_dict = {
        'mode': 'ridesharing',
        'k'   : 20,
        'alpha': 1
    }

    distance_gen_config = {
        'grid_interval'       : 0.05,  # for the rnd distances - how far apart can the agents be in [0, 0.5]
        'num_agents'          : 5,
        'num_steps_training'  : 10,    # how many distances to generate for training
        'revisit_steps'       : 5,     # how many times should distances repeat (to avoid catastrophic forgetting)
        'epsilon'             : 0.01,  # noise on top of the training distances to test generalization
        'num_steps_testing'   : 5 # how many distances to generate
    }


    runner          = ShapleyRunner(setup_dict, char_func_dict, distance_gen_config)

    # TRAIN
    checkpoint_path_ = runner.train()

    # EVALUATE
    # NOTE: The 'compute_action' exploration = False gives better results than True
    #checkpoint_path_ = "/home/zeta/ray_results/shapley_rl/PPO_ShapleyEnv_d52db_00000_0_2023-10-19_14-31-39/checkpoint_000115"

    # 3 agents
   # testing_lst_input =   [[0.46, 0.31, 0.06], [0.5, 0.36, 0.06], [0.5, 0.41, 0.06], [0.5, 0.46, 0.06], [0.46, 0.51, 0.06], [0.41, 0.46, 0.11], [0.5, 0.41, 0.06], [0.46, 0.45, 0.06], [0.5, 0.45, 0.11], [0.5, 0.45, 0.16], [0.46, 0.5, 0.21]]

   # runner.evaluate(checkpoint_path=checkpoint_path_,
   #                  test_distance_lst=testing_lst_input)
    runner.evaluate(checkpoint_path=checkpoint_path_,
                     test_distance_lst=None)
