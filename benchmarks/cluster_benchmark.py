'''Function to calculate the Optimal Coalition Structure via Brute Force
To be used as Ground Truth for the clusters code'''


import numpy as np
from itertools import product, chain


def place_agents(num_agents, line_length, _seed=None):
    """
    Places agents randomly on a line.
    """
    if _seed is not None:
        np.random.seed(_seed)
    return np.random.uniform(0.1, line_length, num_agents)


# Function to compute the value of a coalition
def calculate_coalition_value(coalition, distances, k=20, alpha=1):
    # Filter distances by coalition
    coalition_positions = distances[np.array(coalition) == 1]

    # Calculate value based on the characteristic function
    if len(coalition_positions) == 1:
        return coalition_positions[0]
    if len(coalition_positions) == 0:
        return 0
    else:
        return (k * np.var(coalition_positions) / (alpha * np.sum(coalition)))


# Helper function to get all subsets of a set
def get_subsets(fullset):
    listrep = list(fullset)

    subsets = []
    for i in range(2**len(listrep)):
        subset = []
        for k in range(len(listrep)):
            if i & 1 << k:
                subset.append(listrep[k])
        subsets.append(subset)
    return subsets

# Function to calculate the total value of a coalition structure
def calculate_structure_value(structure, distances, k, alpha):
    total_value = 0
    for coalition in structure:
        if coalition:
            coalition_vector = [1 if i in coalition else 0 for i in range(len(distances))]
            total_value += calculate_coalition_value(coalition_vector, distances, k, alpha)
    return total_value

# Updated function to find the optimal coalition structure
def find_optimal_coalition_structure(distances, k, alpha):
    n_agents = len(distances)
    best_value = np.inf
    best_structure = None

    # Generate all possible subsets of agents
    agent_indices = set(range(n_agents))
    all_subsets = get_subsets(agent_indices)

    # Generate all combinations of these subsets
    for structure in product(all_subsets, repeat=n_agents):
        # Filter out structures with overlapping coalitions
        if len(set(chain(*structure))) != n_agents:
            continue

        structure_value = calculate_structure_value(structure, distances, k, alpha)
        if structure_value < best_value:
            best_value = structure_value
            best_structure = structure

    return best_structure, best_value

def convert_to_one_hot_benchmark(clustering_coalitions, num_agents):
    # Create a set of unique coalitions
    unique_coalitions = set(tuple(sorted(coalition)) for coalition in clustering_coalitions)

    # Initialize one-hot vectors for each unique coalition
    one_hot_clustering = []

    # Fill in the one-hot vectors based on unique coalitions
    for coalition in unique_coalitions:
        if sum(coalition) > 0:  # Skip the empty coalition
            one_hot_vector = [1 if agent in coalition else 0 for agent in range(num_agents)]
            one_hot_clustering.append(tuple(one_hot_vector))

    return one_hot_clustering



if __name__ == '__main__':

    # Example with 3 agents placed randomly in the line [0, 50]
    LINE_LENGTH = 50
    NUM_AGENTS = 3
    SEED =  0

    # Constants for characteristic function
    k = 20
    alpha = 1

    distances = place_agents(NUM_AGENTS, LINE_LENGTH, _seed=SEED)
    rounded_positions = [round(pos, 2) for pos in distances]
    print("Rounded Agent positions:", rounded_positions)

    # Find the optimal coalition structure
    optimal_coalition, optimal_value = find_optimal_coalition_structure(distances, k, alpha)
    optimal_coalition = convert_to_one_hot_benchmark(optimal_coalition,NUM_AGENTS)
    print("Optimal coalition structure:", optimal_coalition)
    print("Value of the optimal coalition:", optimal_value)
