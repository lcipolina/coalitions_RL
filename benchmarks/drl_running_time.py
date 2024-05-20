import matplotlib.pyplot as plt

# Data
number_of_agents = [3, 4, 8, 10]
training_time = [1, 2, 5, 10]

# Create the plot with updated formatting
plt.figure(figsize=(10, 6))
plt.plot(number_of_agents, training_time, marker='o', linestyle='-', color='b', label='DRL (Ours) Method')

# Add title and labels with increased font size
plt.title('Training Time of DRL (Ours) Method vs Number of Agents', fontsize=16)
plt.xlabel('Number of Agents', fontsize=14)
plt.ylabel('Training Time in Compute Hours', fontsize=14)

# Increase tick size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add legend with larger font size
plt.legend(fontsize=12)

# Show the plot
plt.grid(True)
#plt.show()
# Save the plot
plt.savefig('drl_training_time.png')

print("DONE!")