import matplotlib.pyplot as plt

# Your list of distances before and after
distances_before = [[0.45, 0.3, 0.05], [0.5, 0.35, 0.05], [0.5, 0.4, 0.05], [0.5, 0.45, 0.05], [0.45, 0.5, 0.05], [0.4, 0.45, 0.1], [0.5, 0.4, 0.05], [0.45, 0.45, 0.05], [0.5, 0.5, 0.1], [0.5, 0.5, 0.15], [0.45, 0.5, 0.2]]
distances_after = [[0.46, 0.31, 0.06], [0.5, 0.36, 0.06], [0.5, 0.41, 0.06], [0.5, 0.46, 0.06], [0.46, 0.51, 0.06], [0.41, 0.46, 0.11], [0.5, 0.41, 0.06], [0.46, 0.45, 0.06], [0.5, 0.45, 0.11], [0.5, 0.45, 0.16], [0.46, 0.5, 0.21]]

# Initialize empty lists for each agent
agent1_before, agent2_before, agent3_before = [], [], []
agent1_after, agent2_after, agent3_after = [], [], []

# Populate the lists
for d_before, d_after in zip(distances_before, distances_after):
    agent1_before.append(d_before[0])
    agent2_before.append(d_before[1])
    agent3_before.append(d_before[2])

    agent1_after.append(d_after[0])
    agent2_after.append(d_after[1])
    agent3_after.append(d_after[2])

# Create histograms
plt.figure(figsize=(15, 5))

def plot_histogram(subplot_idx, before_data, after_data, agent_name):
    plt.subplot(1, 3, subplot_idx)
    plt.hist(before_data, bins=10, color='#add8e6', alpha=0.99, label='Training distances')  # Faded Blue
    plt.hist(after_data, bins=10, color='#ffcccb', alpha=0.9, label='Testing distances')  # Faded Red
    plt.xlabel('Distance to Origin')
    plt.ylabel('Frequency')
    plt.title(agent_name)
    plt.legend()

plot_histogram(1, agent1_before, agent1_after, 'Agent 1')
plot_histogram(2, agent2_before, agent2_after, 'Agent 2')
plot_histogram(3, agent3_before, agent3_after, 'Agent 3')

plt.tight_layout()

plt.savefig('agent_distances_histogram.png', dpi=300, bbox_inches='tight') # Save the figure
plt.show()
