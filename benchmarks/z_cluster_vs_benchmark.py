import random
import numpy as np
from itertools import product

from juwels.coalitions.Benchmarks.z_clusters import form_coalitions as form_coalitions_clusters
from juwels.coalitions.Benchmarks.z_clusters import unique_coalitions as unique_coalitions_clusters
from juwels.coalitions.Benchmarks.z_clusters import convert_to_one_hot as convert_to_one_hot_clusters
from juwels.coalitions.Benchmarks.z_benchmark import find_optimal_coalition_structure,convert_to_one_hot_benchmark
from juwels.coalitions.Benchmarks.z_benchmark import place_agents as place_agents_benchmark



def compare_coalitions(clustering_coalitions, optimal_coalition_structure):
    # Initialize counter for correct matches
    correct_matches = 0

    # Check each coalition in the clustering result against the optimal structure
    for coalition in clustering_coalitions:
        if coalition in optimal_coalition_structure:
            correct_matches += 1

    # Calculate accuracy
    accuracy = correct_matches / len(clustering_coalitions)
    return accuracy


def run_simulations(num_simulations, num_agents, line_length, epsilon, k, alpha, custom_distances=None):
    total_accuracy = 0
    num_runs = num_simulations if custom_distances is None else min(num_simulations, len(custom_distances))

    for i in range(num_runs):
        print(f"Running simulation {i+1} of {num_runs}")
        # Use custom distances if provided, otherwise generate random distances
        if custom_distances is not None and i < len(custom_distances):
            distances = custom_distances[i]
        else:
            distances = place_agents_benchmark(num_agents, line_length)

        rounded_positions = [round(pos, 2) for pos in distances]
        #print("Rounded Agent positions:", rounded_positions)

        # Clustering method
        clustering_coalitions = unique_coalitions_clusters(form_coalitions_clusters(distances, epsilon))
        one_hot_clustering = convert_to_one_hot_clusters(clustering_coalitions, num_agents)
        #print("One-hot vector representation of clustering coalitions:", one_hot_clustering)

        # Benchmark method
        optimal_coalition, _ = find_optimal_coalition_structure(np.array(distances), k, alpha)
        optimal_coalition = convert_to_one_hot_benchmark(optimal_coalition, num_agents)
        #print("One-hot vector representation of Optimal coalition structure:", optimal_coalition)

        # Compare and calculate accuracy for this simulation
        accuracy = compare_coalitions(one_hot_clustering, optimal_coalition)
        total_accuracy += accuracy
        #print(f"Accuracy of the clustering method: {accuracy * 100:.2f}%")

    # Compute aggregate accuracy
    aggregate_accuracy = total_accuracy / num_runs
    return aggregate_accuracy



#===========================================================================
# RUN THE CODE
# Constants

if __name__ == '__main__':
    # Constants
    LINE_LENGTH = 50
    NUM_AGENTS = 3  #with 5 agents doesn't work. Even in Jewels!
    EPSILON = LINE_LENGTH * 0.05  # Example epsilon value  #63,5%
    EPSILON = LINE_LENGTH * 0.1  # Example epsilon value  #68%
    NUM_SIMULATIONS = 100  # Number of simulations to run
    k = 20  # Scaling constant for characteristic function
    alpha = 1  # Scaling constant for characteristic function
    custom_distances = None

    # Custom distances
    '''
    custom_distances = [
        [20.81, 14.02, 18.16],
        [26.01, 37.26, 44.07],
        [17.14, 34.44, 25.68],
        [43.27, 5.85, 49.72],
        [28.32, 17.04, 8.95]
    ]
    '''
    # Counting the number of elements in the list of lists
    if custom_distances is not None:
        number_of_custom_distances = len(custom_distances)
        num_runs = number_of_custom_distances if custom_distances else NUM_SIMULATIONS
    else:
        num_runs = NUM_SIMULATIONS

    # Run simulations and calculate aggregate accuracy
    aggregate_accuracy = run_simulations(NUM_SIMULATIONS, NUM_AGENTS, LINE_LENGTH, EPSILON, k, alpha, custom_distances=custom_distances)
    print(f"Aggregate accuracy over {num_runs} {'custom runs' if custom_distances else 'simulations'}: {aggregate_accuracy * 100:.2f}%")
