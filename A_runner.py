'''
#!/p/scratch/laionize/cache-kun1/miniconda3/envs/ray_2.6/bin/python
'''

from B_env_shap import ShapleyEnv as Env
from C_train_shap import RunRay as train_policy
from D_deploy import Inference as evaluate_policy
from E_metrics import main as metrics
from E_graphing import graph_reward_n_others
from G_distance_lst import DistanceGenerator

# A_runner.py


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
        self.train_distance_lst = self.distance_gen.generate_training_distances()
        # less observations as we only update every iteration. Num_obs = num_iterations -1
        self.test_distance_lst =self.distance_gen.generate_testing_distances_n_plot(num_obs=None) #same as train dists but moved by an epsilon

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
       # graph_reward_n_others() # generates training graphs avg and rewards per seed - reward, loss, entropy
        return self.checkpoint_path


    def evaluate(self, checkpoint_path=None, test_distance_lst=None):
        '''Reads from checkpoint and plays policyin env. Graphs Boxplot and Accuracy
           To load a checkpoint. Num CPU requested should be the at min, same as training. Otherwise look at open issue in Github.
        '''
        if test_distance_lst is None:
           test_distance_lst=self.test_distance_lst

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        eval_cls = evaluate_policy(checkpoint_path, self.custom_env_config)
        responses_by_distance, final_coalitions_by_distance = eval_cls.play_env_custom_obs(distance_lst_input=test_distance_lst)
        eval_cls.excel_n_plot(responses_by_distance, final_coalitions_by_distance) #plots generalization
        metrics() # reads the summary_table.xls and generates boxplot from the response_data


    def evaluate_custom_dist(self):
        import ast
        '''Passes inputs to the 'evaluate' function above'''
        checkpoint_path_  =  "/p/home/jusers/cipolina-kun1/juwels/ray_results/shapley_rl/PPO_ShapleyEnv_0fad1_00000_0_2024-01-22_20-40-14/checkpoint_000250"
        file_path         = '/p/home/jusers/cipolina-kun1/juwels/coalitions/dist_testing.txt'
        with open(file_path, 'r') as file:
            content = file.read()
        cleaned_content = content.replace('\n,', '\n').replace(',\n', '\n')   # Remove leading commas and extra newlines
        formatted_content = cleaned_content.replace('\n', ', ') # Replace newline characters with commas to form a valid list of lists string
        list_of_lists = ast.literal_eval(formatted_content)     # Convert the string representation of the list of lists into an actual list of lists
        test_distance_lst_ = list_of_lists #[0]                   # Sometimes it comes as tuple
        self.evaluate(checkpoint_path=checkpoint_path_,
                        test_distance_lst=test_distance_lst_)



##################################################
# CONFIGS
##################################################

def run_shapley_runner():
    setup_dict = {
        'training_iterations': 2, #10*25,# (10*25),  this makes a lot of difference!!    # we need ~20 per each distance to learn 100%. Iterations = num_distances * 20
        'train_batch_size'   : 250, #2800,# 2900,   # we need approx 2200 steps to learn 100%
        'seeds_lst'          : [42], #[42,100, 200, 300, 400],#[42,100, 200, 300, 400],
        'experiment_name'    : 'tests_delete', #shapley_rl',
        'cpu_nodes'          : 1 #35 #more than this brakes the custom callbacks (other things work)
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
    #runner.evaluate(checkpoint_path=checkpoint_path_,
    #               test_distance_lst=None)

    #runner.evaluate_custom_dist() #needs the checkpoint!!

if __name__ == '__main__':
    run_shapley_runner()
