from collections import Counter
import matplotlib.pyplot as plt
import os
import ast

current_dir = os.path.dirname(os.path.realpath(__file__))
training_file_path = current_dir+'/dist_training.txt'
testing_file_path   = current_dir+'/dist_testing.txt'

training_file_path = '/p/home/jusers/cipolina-kun1/juwels/coalitions/dist_training_jan31.txt'
testing_file_path  = '/p/home/jusers/cipolina-kun1/juwels/coalitions/dist_testing_jan31_paper.txt'


#========================================================
# Distance "histo graphs" with points
#========================================================



# Helper function
def open_distance_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    # Directly evaluate the string content as a Python expression
    list_of_lists = ast.literal_eval(content)
    return list_of_lists


def plot_agent_training_and_testing_correct_legend(training_file_path, testing_file_path, agent_index):
    """
    Plots the positions of the same agent with training and testing data, using appropriately labeled legends.

    :param training_data: List of lists containing the training data points for each vector.
    :param testing_data: List of lists containing the testing data points for each vector.
    :param agent_index: Index of the agent for both training and testing data.
    """

    # Process training file
    training_distance_lst_ = open_distance_file(training_file_path)
    # Process testing file
    test_distance_lst_ = open_distance_file(testing_file_path)
    # Demonstrating the function with training and testing data for agent 0
    training_data = [[x * 100 for x in sublist] for sublist in training_distance_lst_]
    testing_data = [[x * 100 for x in sublist] for sublist in test_distance_lst_]

    # PLOT
    plt.figure(figsize=(10, 6))

    # Different colors for training and testing
    training_color = 'blue'
    testing_color = 'green'

    # Plotting training data for the specified agent
    training_occurrences = Counter([vector[agent_index] for vector in training_data])
    max_occurrences = max(training_occurrences.values())  # For y-axis scaling

    for distance, count in training_occurrences.items():
        for occurrence in range(count):
            plt.scatter(distance, occurrence + 1, color=training_color)

    # Plotting testing data for the same agent
    testing_occurrences = Counter([vector[agent_index] for vector in testing_data])
    max_testing_occurrences = max(testing_occurrences.values())

    if max_testing_occurrences > max_occurrences:
        max_occurrences = max_testing_occurrences  # Update maximum for y-axis scaling

    for distance, count in testing_occurrences.items():
        for occurrence in range(count):
            plt.scatter(distance, occurrence + 1, color=testing_color)

    # Creating custom legend
    training_legend = plt.Line2D([0], [0], marker='o', color='w', label=f'Agent {agent_index} - training',
                                 markersize=10, markerfacecolor=training_color)
    testing_legend = plt.Line2D([0], [0], marker='o', color='w', label=f'Agent {agent_index} - testing',
                                markersize=10, markerfacecolor=testing_color)

    plt.title(f'Agent {agent_index} Positions with Training and Testing Occurrences')
    plt.xlabel('Agents Location')
    plt.ylabel('Count of Occurrences')
    plt.ylim(0, max_occurrences + 1)  # Adjust y-axis limits
    plt.legend(handles=[training_legend, testing_legend])
    plt.grid(True)
    plt.savefig(current_dir+'/A_results/distances_counter_pol'+ str(agent_index)+'.png')  # Save the figure
    # plt.show()  # Uncomment this line if you want to display the plot as well



#################################################
    #
#####################

# Usage:
for agent in range(0,5):
    plot_agent_training_and_testing_correct_legend(training_file_path, testing_file_path,
                                                    agent_index=agent)
print('done!')