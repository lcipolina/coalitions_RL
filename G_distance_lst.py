'''
When paralelizing environments in RLLIB, each env would generate its own rnd numbers,
then observations are computed and aggregated on the train_batch_size to update the policy
resulting on batches having data from different rnd distances.
Which complicates the learning process.

This pre-computed data is used to train the policy on a fixed set of distances.

The way it works is that we generate a list of distances on a GRID, and then we update each distance
but always staying on the same grid.

Note:

To ensure that each of the 3 agents sees the entire grid of distances based on an interval of 0.05,
you would need a total of 1331 steps--> this should be the minimum number of steps to train on.

This calculation is based on the formula total_steps_needed=num_possible_locations**num_agents
where num_possible_locations is the number of unique distances an agent can be at on the grid.
'''
import datetime
import numpy as np
import random
import os
import matplotlib.pyplot as plt

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M")

# Refactoring the code into a class
class DistanceGenerator:
    def __init__(self, grid_interval=0.05, num_agents=3, num_steps=10, revisit_steps=3, start_value=0.05, epsilon=0.01):
        self.grid_interval = grid_interval
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.revisit_steps = revisit_steps
        self.start_value = start_value
        self.epsilon = epsilon


    def write_to_file(self, distance_list, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))          # Get the directory of the current script
        results_dir = os.path.join(current_dir, 'A_results')              # Construct the full path for the new file
        file_path   = os.path.join(results_dir, filename)
        filename_with_timestamp = f"{file_path}_{TIMESTAMP}.txt"
        with open(filename_with_timestamp, 'w') as file:                  # 'w' mode for writing (overwriting existing data)
            file.write(f"{distance_list}\n")

    def initialize_agent_locations(self):
        possible_locations = [round(x, 2) for x in list(np.arange(self.start_value, 0.51, self.grid_interval))]
        initial_locations = random.sample(possible_locations, self.num_agents)
        return initial_locations

    def generate_training_distance_list(self, initial_locations):
        distance_list = [initial_locations]
        current_locations = initial_locations.copy()

        for i in range(1, self.num_steps): #to generate num_steps-1 distances (as we already have the initial_locations)
            new_locations = []
            for loc in current_locations:
                delta = random.choice([-self.grid_interval, self.grid_interval])
                new_loc = round(loc + delta, 2)
                new_loc = min(max(self.start_value, new_loc), 0.5)
                new_locations.append(new_loc)

            distance_list.append(new_locations)
            current_locations = new_locations.copy()

            if i % self.revisit_steps == 0:
                revisit_location = random.choice(distance_list[:-1])
                distance_list.append(revisit_location)
                current_locations = revisit_location.copy()

        multiplied_data = [[element * 100 for element in sublist] for sublist in distance_list]  # Multiply each number by 100 - to work better on the env
        return multiplied_data

    def generate_training_distances(self):
        initial_locations       = self.initialize_agent_locations()
        self.training_distances = self.generate_training_distance_list(initial_locations)
        self.write_to_file(self.training_distances, filename='training_dist')   # Save to a txt file
        return self.training_distances

    def generate_testing_distances_n_plot(self,train_distance_lst=None ):
        '''Takes the training distances and adds an epsilon - so that tran is fairly close to test'''
        testing_distances = []
        if train_distance_lst is None:                         # If dist list was generated inside this class
           num_obs = len(self.training_distances)              # Loop over the entire training set
        else:
           num_obs = len(train_distance_lst)                   # If dist list was passed.
           self.training_distances = train_distance_lst
        for distance_set in self.training_distances[:num_obs]: # Loop over a slice
        #for distance_set in self.training_distances: #loop over all distances - inlcuding the revisit ones (will be included twice)
                new_set = [round(loc + self.epsilon*100, 2) for loc in distance_set]
                new_set = [min(max(self.start_value, loc), 0.5*100) for loc in new_set]
                testing_distances.append(new_set)
        self.testing_distances = testing_distances

        self.write_to_file(self.testing_distances, filename='testing_dist')   # Save to a txt file

        #self.plot_agent_histograms() # plots the histogram of the training and testing distances

        return self.testing_distances


    def plot_agent_histograms(self):
        # Determine the number of agents from the length of the first sublist
        if self.training_distances and self.testing_distances:
            num_agents = len(self.training_distances[0])
        else:
            raise ValueError("Training or testing distances are empty")

        # Create histograms
        plt.figure(figsize=(5 * num_agents, 5)) #5 inches tall and 5*num_agents inches wide
        for agent_idx in range(num_agents):
            agent_training = [dist[agent_idx] for dist in self.training_distances]
            agent_testing = [dist[agent_idx] for dist in self.testing_distances]

            plt.subplot(1, num_agents, agent_idx + 1)
            plt.hist(agent_training, bins=10, color='#add8e6', alpha=0.99, label='Training Distances')  # Faded Blue
            plt.hist(agent_testing, bins=10, color='#ffcccb', alpha=0.9, label='Testing Distances')  # Faded Red
            plt.xlabel('Distance to Origin')
            plt.ylabel('Frequency')
            plt.title(f'Agent {agent_idx + 1}')
            plt.legend()

        plt.tight_layout()
        script_dir = os.path.dirname(__file__) # Get the directory of the current script
        # Create the full path for the output file
        output_file = os.path.join(script_dir, f'distances_histogram_{TIMESTAMP}.pdf')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save the figure in the same directory as the script


#=========================================================================================
# Test
if __name__ == "__main__":
    distance_gen = DistanceGenerator(grid_interval=0.05, num_agents=3, num_steps=20,
                                     revisit_steps=3,
                                     start_value=0.05,
                                     epsilon=0.01)
    training_distances = distance_gen.generate_training_distances()
    testing_distances = distance_gen.generate_testing_distances_n_plot()
    print("Training Distances:", training_distances)
    print("Testing Distances:", testing_distances)
