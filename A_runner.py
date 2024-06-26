#!/p/scratch/laionize/cache-kun1/miniconda3/envs/ray_2.6/bin/python
# A_runner.py

import ast
from C_train import RunRay as train_policy
from D_deploy import Inference as Inference
from E_metrics import main as metrics
from E_graphing import graph_reward_n_others
from G_distance_lst import DistanceGenerator
import os

script_directory = os.path.dirname(os.path.abspath(__file__))


class CoalitionRunner:
    def __init__(self, setup_dict, char_func_dict, distance_gen_config):
        # Generate list of list of rnd distances and their testing counterparts - they should be generated at the same time to be chained.
        # revisit_steps should be less or equal num_steps

        self.distance_gen_dict = distance_gen_config
        self.setup_dict        = setup_dict
        self.char_func_dict    = char_func_dict

    def set_config_dict_n_dists(self, train_path_= None, test_path_=None):
        ''' Creates the config dict for the gym env. First it needs to build/ receive distances.
        '''
        self.set_distances(train_path= train_path_, test_path=test_path_)  # Training distances are needed to start the env
        # Pass the distance list to the env for CV learning
        self.custom_env_config = {
            'num_agents'      : self.distance_gen_dict['num_agents'],
            'char_func_dict'  : self.char_func_dict,
            'manual_distances': self.train_distance_lst,
            'max_steps'       : 10000, #not used
            'batch_size'      : self.setup_dict['train_batch_size'],
            'cpu_nodes'       : self.setup_dict['cpu_nodes']
        }

    def set_distances(self, train_path= None, test_path=None):
        '''Training and testing distances
           Generated together as they need to be chained
           Or processed from TXT files.
        '''
        self.distance_gen = DistanceGenerator(
        grid_interval = self.distance_gen_dict['grid_interval'],
        num_agents    = self.distance_gen_dict['num_agents'],
        num_steps     = self.distance_gen_dict['num_steps_training'],
        revisit_steps = self.distance_gen_dict['revisit_steps'],
        epsilon       = self.distance_gen_dict['epsilon']
        )
        # TRAIN DISTANCES
        if train_path is None:
           self.train_distance_lst = self.distance_gen.generate_training_distances()
        else:
           self.train_distance_lst = self.open_distance_file(train_path)
        # TEST DISTANCES
        if test_path is None:
           # less nbr of observations as we only update every iteration. Num_obs = num_iterations -1
           # I've commented out the histogram print
           self.test_distance_lst =self.distance_gen.generate_testing_distances_n_plot(train_distance_lst =self.train_distance_lst) #same as train dists but moved by an epsilon

    # RUN THE ENTIRE TRAINING LOOP ====================================
    def train(self,train_path = None,test_path=None):
        '''Runs the entire training loop
           The 'test_path' logic needs to be implemented
        '''
        self.set_config_dict_n_dists(train_path_= train_path, test_path_= test_path) # Training and testing distances and vbles for the env
        # Start Ray to train saves 'output.xlsx'
        train_result         = train_policy(self.setup_dict, self.custom_env_config, self.setup_dict['experiment_name']).train()
        # Reads 'output.xlsx' and generates training graphs avg and rewards per seed - reward, loss, entropy
        graph_reward_n_others()
        return train_result['checkpoint_path'] # used for testing

    # EVALUATE WITH ONE CHECKPOINT ====================================
    def evaluate(self, checkpoint_path=None, train_path= None, test_path = None, max_coalitions_to_plot = 1):
        '''Reads from checkpoint and plays policyin env. Graphs Boxplot and Accuracy
           To load a checkpoint. Num CPU requested should be the at min, same as training. Otherwise look at open issue in Github.
           :inputs: training dists are needed to initialize the env

           OBS: to run the sankey diagram, better to do it from a checkpoint as it takes a lot of time
        '''
        if test_path is None:
           test_distance_lst=self.test_distance_lst
        else:
            test_distance_lst = self.open_distance_file(filepath = test_path)  # Open and process list of list file

        # Training and testing distances and other vbles for the env
        self.set_config_dict_n_dists(train_path_= train_path, test_path_= test_path)    # sets the self.custom_env_config
        eval_cls = Inference(checkpoint_path, self.custom_env_config, self.setup_dict)
        eval_cls.run_inference_and_generate_plots(distance_lst_input=test_distance_lst, max_coalitions=max_coalitions_to_plot) # Run inference on the env and generate coalition plots and 'response_data'
        metrics()                                                                       # Reads the response_data.xls and generates boxplot. Generates the 'summary_table.xls'

    def open_distance_file(self,filepath):
        '''Opens and processes the list of list file'''
        with open(filepath, 'r') as file:
            content = file.read()
        # Directly evaluate the string content as a Python expression
        list_of_lists = ast.literal_eval(content)
        return list_of_lists



##################################################
# CONFIGS
##################################################

def run_coalition_runner(train_n_eval = True, train_path = None,test_path  = None, checkpoint_path_trained = None):
    '''
    Passess the config variables to RLLIB's trainer
    :input: If we want to use a pre-set list of distances - for reproducibility    '''

    #======== Because of the SLURM runner, this needs to be here (otherwise not taken)
    # If we want to use a pre-set list of distances - for reproducibility
    # train_path = os.path.join(script_directory, 'dist_train_jan22.txt')
    # test_path = os.path.join(script_directory, 'dist_test_jan22.txt')

    # TRAIN n EVAL
    #train_n_eval = True

    # EVAL
   # train_n_eval = False # inference only
    #checkpoint_path_trained = \
    #"/p/home/jusers/cipolina-kun1/juwels/ray_results/new_distances/PPO_ShapleyEnv_01c47_00000_0_2024-02-01_15-34-29/checkpoint_000290"
    # =====================

    #TODO: check what's going on with the prints after the training here:
    # 'training_iteration': dfs[next(iter(dfs))]['training_iteration'],
    # MA PERECE QUE LO HABIA ARREGLADO EN EL BRANCH DE JULICH - ver como corre en casa!
    
    setup_dict = {
        'training_iterations': 5, #10*29,# (10*25),  this makes a lot of difference!!    # we need ~20 per each distance to learn 100%. Iterations = num_distances * 20
        'train_batch_size'   :2500, #2900, #2800,# 2900,   # we need approx 2200 steps to learn 100%
        'seeds_lst'          :[42], # [42,100, 200, 300, 400],#[42,100, 200, 300, 400],
        'experiment_name'    :'subadditive_test',
        'cpu_nodes'          : 35 #35 max for Alpha - change it on SLURM  - more than ~38 brakes the custom callbacks (other things work)
    }
    char_func_dict = {
        'mode': 'ridesharing', #'subadditive'
        'k'   :  20,   #### 1,
        'alpha':  1   #### 60,
    }
    distance_gen_config = {
        'grid_interval'       : 0.05,  # for the rnd distances - how far apart can the agents be in [0, 0.5]
        'num_agents'          : 5,     #
        'num_steps_training'  : 10,    # how many distances to generate for training
        'revisit_steps'       : 5,     # how many times should distances repeat (to avoid catastrophic forgetting)
        'epsilon'             : 0.01,  # noise on top of the training distances to test generalization
        'num_steps_testing'   : 5      # how many distances to generate
    }

    # Use the training distances that worked better - diversified points
    runner             = CoalitionRunner(setup_dict, char_func_dict, distance_gen_config)

    if train_n_eval:
        # TRAIN (the 'test_path' logic is TODO)
        checkpoint_path_ = runner.train(train_path = train_path, test_path =test_path )

        # EVALUATE
        # NOTE: The 'compute_action' exploration = False gives better results than True
        runner.evaluate(checkpoint_path = checkpoint_path_,
                        train_path      = train_path,
                        test_path       = test_path,
                        max_coalitions_to_plot = 5) # How many distances to evaluate. To make it faster.
    else: # Evaluate only
        runner.evaluate(checkpoint_path = checkpoint_path_trained,
                        train_path      = train_path,
                        test_path       = test_path,
                        max_coalitions_to_plot = 5) # How many distances to evaluate. To make it faster.


#=====================================
#=====================================

if __name__ == '__main__':

    # If we want to use a pre-set list of distances - for reproducibility
    # OBS: distances were multiplied by 100 to distinguish from no-agent distance (zero)
    train_path = os.path.join(script_directory, 'dist_train_jan22.txt')
    test_path = os.path.join(script_directory, 'dist_test_jan22.txt')
    # ALSO CHANGE inside HERE: run_coalition_runner for "train and eval" or "eval only"

    # TRAIN n Inference
    train_n_eval = True
    checkpoint_path_trained = None

    # EVAL
   #train_n_eval = False # inference only
   # checkpoint_path_trained = \
   # "/Users/lucia/ray_results/subadditive_test/PPO_DynamicCoalitionsEnv_4ec75_00000_0_2024-05-26_15-53-18/checkpoint_000004"
    # =====================


    run_coalition_runner(train_n_eval,
                       train_path = train_path,
                       test_path  = test_path,
                       checkpoint_path_trained =checkpoint_path_trained )
