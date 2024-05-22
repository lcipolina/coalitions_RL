'''Load the saved model and evaluate it in the environment

This one tests how many times an agent has accepted/rejected a coalition

- Generates random coalitions
- Offers them to the agents
- Records the number of times an agent has accepted/rejected a certain coalition
'''
'''
NOTE: in this game, the agent learns to *respond* to the coalitions that are offered to him
This is not a game where the agent learns to *propose* coalitions
We need to offer ALL possible coalitions to the agent to evaluate
The cool thing is:
- The agent does not need to see the characteristic function (at trainig time or at test time)
'''

import torch, socket, os
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go #for graph
import matplotlib.pyplot as plt #for graph
from itertools import permutations
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

current_dir = os.path.dirname(os.path.realpath(__file__))


#Import environment definition
#current_script_dir  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
#parent_dir          = os.path.dirname(current_script_dir) # Get the parent directory (one level up)
#sys.path.insert(0, parent_dir) # Add parent directory to sys.path

# To save results in Juelich
juelich_dir = '/p/scratch/ccstdl/cipolina-kun/A-COALITIONS/'
# Define paths
home_dir = '/Users/lucia/ray_results'
juelich_dir = '/p/scratch/ccstdl/cipolina-kun/A-COALITIONS'
hostname = socket.gethostname() # Determine the correct path
output_dir = juelich_dir if 'jwlogin21.juwels' in hostname.lower() else home_dir
os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = output_dir

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")

from B_env_shap import ShapleyEnv as Env            # custom environment


# ####################################################
# ****************** INFERENCE *********************
# ####################################################

class Inference:

    def __init__(self, checkpoint_path, custom_env_config):
        self.checkpoint_path   = checkpoint_path
        self.custom_env_config = custom_env_config
        self.env               = Env(custom_env_config)
        #self.env_register(env_config=custom_env_config)
        register_env("custom_env", lambda env_ctx: self.env) #the register_env needs a callable/iterable


    def coalition_to_string(self,coalition):
        '''Used for graphing the transitions'''
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return ''.join([alphabet[i] for i, present in enumerate(coalition) if present == 1])

    # Function to convert agent_id to string
    def agent_id_to_string_previous(self,agent_id):
        '''Used for the final coalitions'''
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return alphabet[agent_id]

    def agent_id_to_string(self, agent_id):
        '''Used for the final coalitions'''
        return str(agent_id)



    def generate_observations(self,distance_lst):

        # Select an agent and create all the possible observations for that agent
        # obs_dict = {agent_id: 'coalitions': np.array([c1, c2]), 'distances':np.array([d1, d2]) } # this is how the env expects the observation
        # obs_dict_list of obs_dicts for each agen:
        #  [
        #     {agent_id: 'coalitions':np.array([c1, c2]),'distances':np.array([c1, c2])},   ---> this is how the Policiy expects the observation
        #     {agent_id: 'coalitions':np.array([c1, c2]),'distances':np.array([c1, c2])},.....
        #  ]

        # Generate valid coalitions for each agent
        valid_coalitions_dict = self.env.generate_valid_coalitions()
        # Permute coalitions
        obs_dicts_per_agent = {}
        for agent_id, coalitions in valid_coalitions_dict.items():
            obs_dicts_lst = []
            for c1, c2 in permutations(coalitions, 2):
                obs_dicts_lst.append({agent_id: {'coalitions':np.vstack([c1, c2]),
                                                 'distances':np.vstack(np.array([c1*distance_lst, c2*distance_lst]))
                                                    }
                                        })
            obs_dicts_per_agent[agent_id] = obs_dicts_lst
        return obs_dicts_per_agent


    def play_env_custom_obs(self, distance_lst_input=None):
        # Initialize Ray
        if ray.is_initialized(): ray.shutdown()
        #ray.init(local_mode=False, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
        #ray.init(address='auto',include_dashboard=False, ignore_reinit_error=True,log_to_driver=True, _temp_dir = '/p/home/jusers/cipolina-kun1/juwels/ray_tmp')

        # Rebuild the policy
        algo = Algorithm.from_checkpoint(self.checkpoint_path)

        # Initialize variables
        responses_by_distance = {}  # New dictionary to store responses by distance list
        final_coalitions_by_distance = {}  # New dictionary to store final coalitions by distance list

        max_steps = self.custom_env_config['max_steps']

        # Set default distances if none are provided
        if distance_lst_input is None:
           distance_lst_input =  [self.env.distance_lst] #take *one* distance from the env (rnd generated)

        for distance_lst in distance_lst_input:
            self.distance_lst = distance_lst
            print('env distance_lst:', distance_lst)

            # _, _ = self.env.reset()

            # Generate all permutations of coalitions and add distances to the dict
            obs_dicts_per_agent = self.generate_observations(distance_lst) # creates{[coalitions], [distances]}
            response_lst        = []    # Initialize list to store responses for this distance list
            final_coalitions    = set() # Initialize set to store final coalitions for this distance list


            # Run the environment on the trained model
            for agent_id, obs_dict_lst in obs_dicts_per_agent.items():
                for obs_dict in obs_dict_lst:

                    agent_id = list(obs_dict.keys())[0]
                    policy_agent = "policy" + str(agent_id)
                    current_coal, proposed_coal = obs_dict[agent_id]['coalitions'] #used for the reward

                   # print('obs_dict', obs_dict)

                    # Compute action
                    action, states, extras_dict = algo.get_policy(policy_id=policy_agent).compute_single_action(obs_dict, explore=False)

                    # Get reward
                    rew_current_agent = self.env._calculate_reward(action, agent_id, current_coal, proposed_coal, distance_lst=distance_lst, mode=self.custom_env_config['char_func_dict']['mode'])

                    # Save results
                    response_entry = {'agent_id': agent_id, 'current': current_coal, 'proposed': proposed_coal, 'action': action, 'rew': rew_current_agent}
                    response_lst.append(response_entry)

                    # Check if the proposed coalition is accepted by all its members
                    if action == 1 and sum(proposed_coal):
                        proposed_coal_str = self.coalition_to_string(proposed_coal)
                        final_coalitions.add(proposed_coal_str)

            # Store the responses and final coalitions for this distance list
            distance_str = str([round(x, 2) for x in distance_lst])
            responses_by_distance[distance_str] = response_lst
            final_coalitions_by_distance[distance_str] = list(final_coalitions)

        ray.shutdown()
        return responses_by_distance, final_coalitions_by_distance


    def excel_n_plot(self,responses_by_distance, final_coalitions_by_distance):
        self.save_to_excel(responses_by_distance)                                        # saves the data to an excel file
        #self.sankey_diagram(responses_by_distance, final_coalitions_by_distance)   # plots the dynamics of the game
        self.plot_final_coalitions( responses_by_distance, final_coalitions_by_distance) # plots the final coalitions


    def save_to_excel(self, responses_by_distance):
        '''Save the DataFrame to an Excel file
        responses_by_distance is a dictionary where keys are distance lists and values are lists of dictionaries
        '''
        #with pd.ExcelWriter('response_data.xlsx') as writer:
        with pd.ExcelWriter(current_dir+'/A_results/response_data.xlsx') as writer:
            for distance, response in responses_by_distance.items():
                df = pd.DataFrame(response)
                df['current'] = df['current'].apply(list)  # Convert NumPy arrays to lists (if they are NumPy arrays)
                df['proposed'] = df['proposed'].apply(list)

                # Sanitize sheet name by replacing invalid characters
                sanitized_sheet_name = str(distance).replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")

                df.to_excel(writer, sheet_name=sanitized_sheet_name, index=False)


    def plot_final_coalitions(self, responses_by_distance, final_coalitions_by_distance):
        '''Plot the accepted coalitions for each distance list'''

        plt.figure(figsize=(15, 10))

        # Initialize color map
        colors = ['red', 'green', 'blue', 'purple', 'orange']

        for idx, (distance_str, response_lst) in enumerate(responses_by_distance.items()):
            # Create a dictionary to store the last coalition for each agent
            last_coalitions = {}
            for entry in response_lst:
                if entry['action'] == 1:
                    last_coalitions[entry['agent_id']] = entry['proposed']

            color_mapping = {}

            # Annotate each point and draw connecting lines for coalitions
            distance_lst = [round(x*100, 2) for x in eval(distance_str)]
            for i, distance in enumerate(distance_lst):
                agent_id = i
                coalition_array = tuple(last_coalitions.get(agent_id, []))

                if coalition_array not in color_mapping:
                    color_mapping[coalition_array] = colors[len(color_mapping) % len(colors)]

                color = color_mapping[coalition_array]
                coalition_str = self.agent_id_to_string(agent_id)

                plt.scatter(distance, idx, c=color, marker='o')
                plt.annotate(coalition_str, (distance, idx), textcoords="offset points", xytext=(0, 10), ha='center')

                for j in range(i + 1, len(distance_lst)):
                    if np.array_equal(last_coalitions.get(agent_id, []), last_coalitions.get(j, [])):
                        plt.plot([distance, distance_lst[j]], [idx, idx], color=color)

        plt.xlabel('Distance from Origin', fontsize=14)
        plt.ylabel('Game Index', fontsize=14)
        plt.title('Final Coalitions for Each Distance Game', fontsize=16)
        plt.yticks(range(len(responses_by_distance)))  # Show y-axis ticks for each run
        plt.xlim(0, 0.6*100)  # Set x-axis limits
        plt.savefig(current_dir+'/A_results/final_coals_graph_' +TIMESTAMP+'.pdf')
       # plt.show()

    def sankey_diagram(self, responses_by_distance, final_coalitions_by_distance):
        '''responses_by_distance and final_coalitions_by_distance are dictionaries with distance lists as keys'''
        for distance_lst, response_lst in responses_by_distance.items():
            self.plot_sankey_diagram(response_lst, distance_lst)

    def plot_sankey_diagram(self, response_lst, distance_lst):
        # Diagram
        labels, sources, targets, values, colors = [], [], [], [], []
        # Custom colors
        color_n = ['#55CBCD', '#CBAACB', '#FF968A', '#4DD091', '#FF5768', '#0065A2', '#57838D', '#FFC500']  # node colors - more vibrant
        color_l = ['#D4F0F0', '#ECD5E3', '#FFDBCC', '#E0F8F5', '#FFEFFF', '#9EDDEF', '#D7D2EA', '#FFF7C2']  # link colors - softer

        # Populate data structures
        for response in response_lst:
            if response['action'] == 1:  # Only consider accepted coalitions
                current_str = self.coalition_to_string(response['current'])
                proposed_str = self.coalition_to_string(response['proposed'])

                if current_str not in labels:
                    labels.append(current_str)
                if proposed_str not in labels:
                    labels.append(proposed_str)

                sources.append(labels.index(current_str))
                targets.append(labels.index(proposed_str))
                values.append(1)  # Assuming each transition has a value of 1

                # Assign colors to nodes and links
                agent_color = color_l[response['agent_id'] % len(color_l)]  # cycle through the colors to create more
                colors.append(agent_color)

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=labels,
                color=color_n[:len(labels)]  # Set node colors
            ),
            link=dict(source=sources, target=targets, value=values, color=colors  # Set the link colors
            )
        )])

        # Add title to indicate which distance list this diagram corresponds to
        fig.update_layout(title_text=f"Sankey Diagram for Distance List: {distance_lst}")

        # Save the figure to a PDF file
        sanitized_distance_str = str(distance_lst).replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
        fig.write_image(f"sankey_diagram_{sanitized_distance_str}.pdf")


###############################################
# MAIN
################################################
if __name__=='__main__':


    # SETUP for env and model --> MAKE SURE IT MATCHES THE TRAINING CODE !!
    char_func_dict = {
        'mode'  : 'ridesharing', #'closed_form',
        'k'     : 100,  # k = N^2 for it to be non-superadditive
        'alpha' : 1, # alpha = 1 for it to be subadditive
    }

    custom_env_config = {
                'num_agents'     : 4,
                'char_func_dict' : char_func_dict,
                'max_steps'      : 8000,
                 'batch_size'     : 1000 # for the CV learning - one update per batch size
                }

    cls = Inference(Env,checkpoint_path, custom_env_config)

    # CHOOSE ONE
    #evaluate(custom_env_config) # this one gives some wrong actions for the same observations
    #previous_code(custom_env_config) # this one gives the right actions
    response_lst = cls.play_env()  # Environment creates the observations
   #  eval_cls.excel_n_plot(responses_by_distance, final_coalitions_by_distance )



    '''
    #This is to show that the policy for agent 2 is indeed doing the right thing
    # I don't know why it comes wrong on the other code
    agent_id = 2
    a = {agent_id :np.array([[1, 0, 1],[1, 1, 1]])}, # esto da 0 en el otro codigo pero aca da 1
    b = {agent_id :np.array([[1, 1, 1],[1, 0, 1]])} #esto da 1 en el otro codigo pero aca da 0
    c = {agent_id :np.array([[0, 1, 1],[1, 1, 1]])} #esto da 0 en el otro codigo pero aca da 1

    action, states, extras_dict = algo.get_policy(policy_id="policy"+str(agent_id)).compute_single_action(a, explore = True) #explore = true samples from the action distribution while False takes the mode (a fixed value)
    print('a', action)
    action, states, extras_dict = algo.get_policy(policy_id=policy_agent).compute_single_action(b, explore = True) #explore = true samples from the action distribution while False takes the mode (a fixed value)
    print('b', action)
    action, states, extras_dict = algo.get_policy(policy_id=policy_agent).compute_single_action(c, explore = True) #explore = true samples from the action distribution while False takes the mode (a fixed value)
    print('b', action)
    '''
