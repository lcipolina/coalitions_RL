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

import numpy as np
import random
import logging
logging.basicConfig(filename='distance_log.log', level=logging.INFO, format='%(message)s')


# Refactoring the code into a class
class DistanceGenerator:
    def __init__(self, grid_interval=0.05, num_agents=3, num_steps=10, revisit_steps=3, start_value=0.05, epsilon=0.01):
        self.grid_interval = grid_interval
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.revisit_steps = revisit_steps
        self.start_value = start_value
        self.epsilon = epsilon
        self.training_distances = self.generate_training_distances()
        self.testing_distances = self.generate_testing_distances()

    def initialize_agent_locations(self):
        possible_locations = [round(x, 2) for x in list(np.arange(self.start_value, 0.51, self.grid_interval))]
        initial_locations = random.sample(possible_locations, self.num_agents)
        return initial_locations

    def generate_distance_list(self, initial_locations):
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

        # Save to a logging text file - note that RLLIB resets the env several times before training for real
        log_entry = f" training distances: {distance_list}"
        logging.info(log_entry)

        return distance_list

    def generate_training_distances(self):
        initial_locations = self.initialize_agent_locations()
        return self.generate_distance_list(initial_locations)

    def generate_testing_distances(self,num_obs = None ):
        testing_distances = []
        if num_obs is None: #loop over the entire training set
            num_obs = len(self.training_distances)

        for distance_set in self.training_distances[:num_obs]: #loop over a slice
                new_set = [round(loc + self.epsilon, 2) for loc in distance_set]
                new_set = [min(max(self.start_value, loc), 0.5) for loc in new_set]
                testing_distances.append(new_set)
        return testing_distances

# Test the class under a __main__ guard
if __name__ == "__main__":
    distance_gen = DistanceGenerator(grid_interval=0.05, num_agents=3, num_steps=20, revisit_steps=3, epsilon=0.01)
    print("Training Distances:", distance_gen.training_distances)
    print("Testing Distances:", distance_gen.testing_distances)

    distance_gen.generate_training_distances
