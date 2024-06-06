'''Load the saved model and evaluate it in the environment
 Saves a file: response_data_TIME_STAMP.xlsx which is used to generate the boxplot and the summary table by the Metrics class


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
import plotly.graph_objects as go #for sankey diagram
import matplotlib.pyplot as plt
from itertools import permutations
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from C_ppo_config import get_marl_trainer_config

current_dir = os.path.dirname(os.path.realpath(__file__))


#Import environment definition
#current_script_dir  = os.path.dirname(os.path.realpath(__file__)) # Get the current script directory path
#parent_dir          = os.path.dirname(current_script_dir) # Get the parent directory (one level up)
#sys.path.insert(0, parent_dir) # Add parent directory to sys.path

# To save results in Juelich
juelich_dir = '/p/scratch/ccstdl/cipolina-kun/A-COALITIONS/'
# Define paths
home_dir = '/Users/lucia/ray_results'
hostname = socket.gethostname() # Determine the correct path
output_dir = juelich_dir if 'jwlogin21.juwels' in hostname.lower() else home_dir
os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = output_dir

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")
response_file = current_dir+'/A_results/response_data_'+TIMESTAMP+'.xlsx'  #responses and final coalitions


from B_env import DynamicCoalitionsEnv as Env            # custom environment


class Inference:

    def __init__(self, checkpoint_path, custom_env_config, setup_dict):
        self.checkpoint_path   = checkpoint_path
        self.custom_env_config = custom_env_config
        self.setup_dict        = setup_dict
        self.env               = Env(custom_env_config)
        register_env("custom_env", lambda env_ctx: self.env) #the register_env needs a callable/iterable

    def coalition_to_string(self,coalition):
        '''Used for graphing the transitions'''
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return ''.join([alphabet[i] for i, present in enumerate(coalition) if present == 1])

    def agent_id_to_string(self, agent_id):
        '''Used for the final coalitions'''
        return str(agent_id)

    #==========================================================
    # Step 1 - Run the environment and collect responses and coalitions
    #==========================================================

    def generate_observations(self,distance_lst):
        '''Generate all possible coalitions and distances for each agent
         Select an agent and create all the possible observations for that agent
         obs_dict = {agent_id: 'coalitions': np.array([c1, c2]), 'distances':np.array([d1, d2]) } # this is how the env expects the observation
         obs_dict_list of obs_dicts for each agen:
          [
             {agent_id: 'coalitions':np.array([c1, c2]),'distances':np.array([c1, c2])},   ---> this is how the Policiy expects the observation
             {agent_id: 'coalitions':np.array([c1, c2]),'distances':np.array([c1, c2])},.....
          ]
        '''
        valid_coalitions_dict = self.env.generate_valid_coalitions()   # Generate valid coalitions for each agent
        obs_dicts_per_agent = {}
        for agent_id, coalitions in valid_coalitions_dict.items(): # Permute coalitions
            obs_dicts_lst = []
            for c1, c2 in permutations(coalitions, 2):
                obs_dicts_lst.append({agent_id: {'coalitions':np.vstack([c1, c2]),
                                                 'distances':np.vstack(np.array([c1*distance_lst, c2*distance_lst]))
                                                    }     })
            obs_dicts_per_agent[agent_id] = obs_dicts_lst
        return obs_dicts_per_agent

    def play_env_custom_obs(self, distance_lst_input=None):
        '''Play the environment with the trained model and generate responses and final coalitions
        '''
        if ray.is_initialized(): ray.shutdown()
        ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
        #ray.init(address='auto',include_dashboard=False, ignore_reinit_error=True,log_to_driver=True, _temp_dir = '/p/home/jusers/cipolina-kun1/juwels/ray_tmp')

        # Inference with less number of CPUs Convert checkpoint info to algorithm state
        checkpoint_info = get_checkpoint_info(self.checkpoint_path)
        state = Algorithm._checkpoint_info_to_algorithm_state(
            checkpoint_info=checkpoint_info,
            policy_ids=None,                             # Adjust if using specific policy IDs; might be auto-inferred in your case
            policy_mapping_fn=None,                      # Set to None as we'll configure policies directly in the config
            policies_to_train=None,                      # Specify if training specific policies
        )

        # Need to bring the config dict exactly as it was when training (otherwise won't work)
        self.setup_dict['cpu_nodes'] = 7                 # Modify the configuration for your multi-agent setup and 7 CPU cores
        self.setup_dict['seed']      = 42                # Because the setup_dict actually contains a lst, not a single value
        modified_config  = get_marl_trainer_config(Env, self.custom_env_config, self.setup_dict)
        state["config"]  = modified_config.to_dict()     # Inject the modified configuration into the state
        algo = Algorithm.from_state(state)               # Load the algorithm from the modified state

        # Rebuild the policy - old way
        #algo = Algorithm.from_checkpoint(self.checkpoint_path)

        # Initialize variables
        responses_by_distance  = {}            # Dict to store responses by distance list
        accepted_coalitions_by_distance = {}    # Dict to store accepted coalitions by distance list

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
            accepted_coalitions = set() # Initialize set to store final coalitions for this distance list

            # Run the environment on the trained model
            for agent_id, obs_dict_lst in obs_dicts_per_agent.items():
                for obs_dict in obs_dict_lst:

                    agent_id = list(obs_dict.keys())[0]
                    policy_agent = "policy" + str(agent_id)
                    current_coal, proposed_coal = obs_dict[agent_id]['coalitions'] #used for the reward

                   # print('obs_dict', obs_dict)

                    # Compute action - This way needs flattened obs dict
                    action, states, extras_dict = algo.get_policy(policy_id=policy_agent).compute_single_action(obs_dict, explore=False)

                    # Get reward
                    reward_current_agent = self.env._calculate_reward(action, agent_id, current_coal, proposed_coal, distance_lst=distance_lst, mode=self.custom_env_config['char_func_dict']['mode'])

                    # Save results
                    response_entry = {'agent_id': agent_id, 'current': current_coal, 'proposed': proposed_coal, 'action': action, 'rew': reward_current_agent}
                    response_lst.append(response_entry)

                    # Store accepted coalitions (if they are not empty) - doesn't store which agent accepted it - just that it was accepted
                    if action == 1 and sum(proposed_coal):
                        proposed_coal_str = self.coalition_to_string(proposed_coal)
                        accepted_coalitions.add(proposed_coal_str)

            # Store the responses and final coalitions for this distance list
            distance_str = str([round(x, 2) for x in distance_lst])
            responses_by_distance[distance_str] = response_lst                        # Which agent accepted/rejected which coalition
            accepted_coalitions_by_distance[distance_str] = list(accepted_coalitions) # All the coalitions accepted for this distance list (don't store which agent accepted it)

        ray.shutdown()
        return responses_by_distance, accepted_coalitions_by_distance

    #==========================================================
    # Step 2 - Saving and plotting results
    #==========================================================

    def save_results_to_excel(self, responses_by_distance, accepted_coalitions_by_distance):
        '''Save the responses and final coalitions to an Excel file'''
        responses_df = self.convert_responses_to_dataframe(responses_by_distance)
        coalitions_df = self.convert_coalitions_to_dataframe(accepted_coalitions_by_distance)
        with pd.ExcelWriter(response_file) as writer:
            responses_df.to_excel(writer, sheet_name='Responses')


    def convert_responses_to_dataframe(self, responses_by_distance):
        '''Convert the responses dictionary to a pandas DataFrame'''
        data = []
        for distance_str, responses in responses_by_distance.items():
            for response in responses:
                data.append({
                    'Distance': distance_str,
                    'agent_id': response['agent_id'],  #read with this label by the 'metrics' script
                    'Current Coalition': self.coalition_to_string(response['current']),
                    'Proposed Coalition': self.coalition_to_string(response['proposed']),
                    'Action': response['action'],
                    'rew': response['rew']             #read with this label by the 'metrics' script
                })
        return pd.DataFrame(data)

    def convert_coalitions_to_dataframe(self, accepted_coalitions_by_distance):
        '''Convert the final coalitions dictionary to a pandas DataFrame'''
        data = []
        for distance_str, coalitions in accepted_coalitions_by_distance.items():
            for coalition in coalitions:
                data.append({
                    'Distance': distance_str,
                    'Final Coalition': coalition
                })
        return pd.DataFrame(data)

    def plot_final_coalitions(self, responses_by_distance, accepted_coalitions_by_distance):
        '''Plot the accepted coalitions for each distance list
           Loops through every distance, agent and response to plot the final coalitions
           Gets the last accepted proposed coalition for each agent for each distance
           :responses_by_distance: is a dictionary with distance lists as keys and values are lists of responses by all agents
           :accepted_coalitions_by_distance: is a dictionary with distance lists as keys and values are lists of accepted coalitions - it doesn't store which agent accepted it - so not very useful for plotting
        '''

        plt.figure(figsize=(15, 10))
        colors = ['red', 'green', 'blue', 'purple', 'orange']  # Initialize color map

        # Iterate over each distance list
        for agent_idx, (distance_str, response_lst) in enumerate(responses_by_distance.items()):
            last_coalitions = {}                                                        # Dict to store the last coalition for each agent
            distance_lst = [round(x, 2) for x in eval(distance_str)]                    # Convert distance string to list
            accepted_coalitions = accepted_coalitions_by_distance.get(distance_str, []) # Use accepted_coalitions_by_distance to get the final coalitions directly
            color_mapping = {}                                                          # Dict to store color mapping for each coalition tuple

            # Iterate over each response to get the last proposed coalition accepted by each agent for this distance - 'response_lst' is a list of responses for each agent and this distance list
            for entry in response_lst:                                                  # Iterate over each response for this distance list and update the last coalition for each agent
                if entry['action'] == 1:                                                # Only consider accepted proposed coalitions
                    last_coalitions[entry['agent_id']] = entry['proposed']              # Update the last coalition for this agent with the proposed coalition

            # Iterate over each agent to plot the final coalitions accepted by each agent
            for i, distance in enumerate(distance_lst):                                 # Iterate over each agent and distance_coordinate
                agent_id = i                                                            # Agent ID is the index
                coalition_array = tuple([1 if chr(i + 65) in accepted_coalitions else 0 for i in range(len(distance_lst))]) # Create a tuple of 1s and 0s to represent the coalition - convert from A, B, C to 0, 1, 2

                # Manage color mapping
                if coalition_array not in color_mapping:                                      # Assign a color to each coalition. Coals are tuple s and colors are strings, the mapping is a dict with the tuple as key and the color as value (the color is a string)
                    color_mapping[coalition_array] = colors[len(color_mapping) % len(colors)] # Cycle through the colors to create more
                color = color_mapping[coalition_array]
                coalition_str = self.agent_id_to_string(agent_id)

                plt.scatter(distance, agent_idx, c=color, marker='o')                         # Plot the agent at the distance
                plt.annotate(coalition_str, (distance, agent_idx), textcoords="offset points", xytext=(0, 10), ha='center')

                # Connect agents with the same coalition
                for j in range(i + 1, len(distance_lst)):                                             # Iterate over the remaining agents - to ensure we don't double count agents
                    if np.array_equal(last_coalitions.get(agent_id, []), last_coalitions.get(j, [])): # Check if the current agent (agent_id) and the agent with index j are part of the same coalition.
                        plt.plot([distance, distance_lst[j]], [agent_idx, agent_idx], color=color)    # Plot the connection - 'agent_idx' is the y-coordinate of the agent, distance is the x-coordinate

        plt.xlabel('Distance from Origin', fontsize=14)
        plt.ylabel('Game Index', fontsize=14)
        plt.title('Final Coalitions for Each Distance Game', fontsize=16)
        plt.yticks(range(len(responses_by_distance)))  # Show y-axis ticks for each run
        plt.xlim(0, 60)  # Set x-axis limits (assuming distance is scaled to percentage)
        plt.savefig(current_dir+'/A_results/final_coals_graph_' + TIMESTAMP + '.png')

    def sankey_diagram(self, responses_by_distance):
        ''' Visualize the transitions between coalitions
            :responses_by_distance: and :accepted_coalitions_by_distance: are dictionaries with distance lists as keys'''
        for distance_lst, response_lst in responses_by_distance.items(): # get all responses for each distance list
            self.plot_sankey_diagram(response_lst, distance_lst)         # Plot the Sankey diagram with all the accepted coalitions for this distance list

    def plot_sankey_diagram(self, response_lst, distance_lst):
        '''Plot a Sankey diagram for the accepted coalitions for a given distance list
           :response_lst: is a list of responses for each agent and this distance list: [{'agent_id': agent_id, 'current': current_coal, 'proposed': proposed_coal, 'action': action, 'rew': reward_current_agent}]
           :distance_lst: is the distance list for which the Sankey diagram is being plotted - only used for the title
        '''
        labels, sources, targets, values, colors = [], [], [], [], []
        label_indices = {}  # Dictionary to store unique values for each node
        transition_counts = {}  # Dictionary to track transition counts

        # Custom colors
        color_n = ['#55CBCD', '#CBAACB', '#FF968A', '#4DD091', '#FF5768', '#0065A2', '#57838D', '#FFC500']  # node colors - more vibrant
        color_l = ['#D4F0F0', '#ECD5E3', '#FFDBCC', '#E0F8F5', '#FFEFFF', '#9EDDEF', '#D7D2EA', '#FFF7C2']  # link colors - softer

        # Iterate over each response to get the current and proposed coalitions
        # and create the nodes and links for the Sankey diagram
        for response in response_lst:  # Iterate over each response for this distance list
            if response['action'] == 1:  # Only consider accepted coalitions
                current_str = self.coalition_to_string(response['current']) #TODO: this converts to a string - it might not be needed - Agents as numbers are fine
                proposed_str = self.coalition_to_string(response['proposed']) #TODO: this takes

                # Add the current and proposed coalitions to the labels list if they are not already present - each node is a coalition and should be represented only once
                if current_str not in label_indices:
                    label_indices[current_str] = len(labels)
                    labels.append(current_str)

                # Add proposed_str to labels if not already present
                if proposed_str not in label_indices:
                    label_indices[proposed_str] = len(labels)
                    labels.append(proposed_str)

                # Get indices for sources and targets
                source_idx = label_indices[current_str]
                target_idx = label_indices[proposed_str]

                # Aggregate transition counts
                 #multiple agents can transition from the same origin to the same target coalition, the Sankey diagram needs to account for the aggregated transitions.
                 #   Specifically, we need to sum the transitions between the same source and target rather than duplicating them.
                if (source_idx, target_idx) not in transition_counts: # Initialize the transition count to 0 if it doesn't exist
                    transition_counts[(source_idx, target_idx)] = 0
                transition_counts[(source_idx, target_idx)] += 1

                 # Assign colors to nodes and links (same color for multiple transitions for simplicity)
                agent_color = color_l[response['agent_id'] % len(color_l)]  # Cycle through the colors to create more
                colors.append(agent_color)

        # Populate sources, targets, and values based on aggregated transitions
        for (source_idx, target_idx), count in transition_counts.items():
            sources.append(source_idx) # Starting node of each transition
            targets.append(target_idx) # Ending node of each transition
            values.append(count) # Value of the transition

         # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=labels,
                color=color_n[:len(labels)]  # Set node colors
            ),
            link=dict(source=sources, target=targets, value=values, color=colors[:len(sources)]  # Set the link colors
            )
        )])

        # Add title to indicate which distance list this diagram corresponds to
        fig.update_layout(title_text=f"Sankey Diagram for Distance List: {distance_lst}")

        # Save the figure to a PNG file
        sanitized_distance_str = str(distance_lst).replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")  # For the title
        fig.update_layout(autosize=False, width=800, height=600)  # Resize to make it smaller
        fig.write_image(f"sankey_diagram_{sanitized_distance_str}.png", scale=1, width=800, height=600)



    #================================================
    # RUN INFERENCE AND GENERATE PLOTS
    #================================================
    def run_inference_and_generate_plots(self, distance_lst_input=None,max_coalitions=None):
        '''Run inference on the environment and generate coalition plots and 'response_data'
           :distance_lst_input: is a list of distance lists to be used in the environment
           :max_coalitions: is the maximum number of coalitions to be generated for each agent'''

        if max_coalitions is not None: # Trim the distance list to make this more manageable
            distance_lst_input = distance_lst_input[:max_coalitions]

        responses_by_distance, accepted_coalitions_by_distance = self.play_env_custom_obs(distance_lst_input) # The first return stores the responses of all agents and the second stores the accepted coalitions
        self.plot_final_coalitions(responses_by_distance, accepted_coalitions_by_distance)
        #self.sankey_diagram(responses_by_distance) # takes forever to save if there are multiple nodes.
        self.save_results_to_excel(responses_by_distance, accepted_coalitions_by_distance)  # Assuming this method exists for saving results to Excel



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

    cls = Inference(output_dir, custom_env_config)

    # CHOOSE ONE
    #evaluate(custom_env_config) # this one gives some wrong actions for the same observations
    #previous_code(custom_env_config) # this one gives the right actions
    response_lst = cls.play_env()  # Environment creates the observations
   #  eval_cls.excel_n_plot(responses_by_distance, accepted_coalitions_by_distance )



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
