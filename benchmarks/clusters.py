import matplotlib.pyplot as plt
import random



def form_coalitions(distances, epsilon):
    """
    Forms coalitions based on the epsilon distance.
    Returns:
    [[0, 1], [0, 1], [2], [3, 4], [3, 4]]    which means, which coalition each agent
    """
    coalitions = []
    for i, agent in enumerate(distances):
        current_coalition = [j for j, other_agent in enumerate(distances) if abs(agent - other_agent) <= epsilon]
        coalitions.append(current_coalition)
    return coalitions

def unique_coalitions(coalitions):
    """
    Transforms the coalitions list to a list of unique coalitions.
    """
    unique_coalitions = []
    for coalition in coalitions:
        # Sort and convert to tuple for uniqueness
        sorted_coalition = tuple(sorted(coalition))
        if sorted_coalition not in unique_coalitions:
            unique_coalitions.append(sorted_coalition)
    return unique_coalitions

def convert_to_one_hot(clustering_coalitions, num_agents):
    # Create a set of unique coalitions
    unique_coalitions = set(tuple(sorted(coalition)) for coalition in clustering_coalitions)

    # Initialize one-hot vectors for each unique coalition
    one_hot_clustering = []

    # Fill in the one-hot vectors based on unique coalitions
    for coalition in unique_coalitions:
        one_hot_vector = [0] * num_agents
        for agent in coalition:
            one_hot_vector[agent] = 1
        one_hot_clustering.append(tuple(one_hot_vector))

    return one_hot_clustering

def place_agents(num_agents, line_length, _seed=None):
    """
    Places agents randomly on a line.
    """
    if _seed is not None:
        np.random.seed(_seed)
    return np.random.uniform(0.1, line_length, num_agents)

# PLOT============================
def plot_coalitions(agents, coalitions, line_length):
    """
    Plots agents and their coalitions.
    """
    plt.figure(figsize=(10, 2))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, agent in enumerate(agents):
        plt.plot(agent, 1, 'o', color=colors[i % len(colors)], label=f'Agent {i+1}')

    for coalition in coalitions:
        if len(coalition) > 1:
            for member in coalition:
                plt.plot([agents[member], agents[coalition[0]]], [1, 1], linestyle='-', color=colors[coalition[0] % len(colors)])

    plt.xlim(0, line_length)
    plt.ylim(0.5, 1.5)
    plt.title('Spatial Coalition Game')
    plt.xlabel('Position on Line')
    plt.legend()
    plt.show()


# RUN================================================
if __name__ == '__main__':
    # Constants
    LINE_LENGTH = 50
    NUM_AGENTS = 5
    EPSILON = LINE_LENGTH*0.1  # Example epsilon value
    SEED = 0

    # Place agents and form coalitions
    distances = place_agents(NUM_AGENTS, LINE_LENGTH, _seed=SEED)
    coalitions = form_coalitions(distances, EPSILON)

    # Transform to a list of unique coalitions
    unique_coalitions = unique_coalitions(coalitions)

    print("Agents' positions:", distances)
    print("Coalitions formed:", unique_coalitions)

    one_hot_clustering = convert_to_one_hot(unique_coalitions, NUM_AGENTS)
    print("One-hot vector representation of clustering coalitions:", one_hot_clustering)

    # Plot the coalitions
    #plot_coalitions(distances, coalitions, LINE_LENGTH)
