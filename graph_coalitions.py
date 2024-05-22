
''' Script to replicate the graph of the coalitions in the paper as I can't find the original ones'''


import matplotlib.pyplot as plt

# Red data points adjusted across the 0 to 50 x-axis range with specific labels
red_data_old = [
    ([5, 11], [8, 8], ['0', '1']),
    ([13, 15], [7, 7], ['1', '0']),
    ([10, 15], [6, 6], ['1', '0']),
    ([5, 15], [5, 5], ['0', '1']),
    ([7, 11], [4, 4], ['0', '1']),
    ([6, 10], [3, 3], ['1', '0']),
    ([8, 10], [2, 2], ['1', '0']),
    ([8, 11], [1, 1], ['0', '1']),
    ([8, 12], [0, 0], ['1', '0']),
]

# Green data points
green_data_old = [
    ([30, 35, 45], [8, 8, 8], ['3', '2', '4']),
    ([38, 40, 47], [7, 7, 7], ['4', '2', '3']),
    ([33], [6], ['2']),
    ([40, 42, 48], [5, 5, 5], ['4', '2', '3']),
    ([36, 42], [4, 4], ['3', '2']),
    ([38, 45, 48], [3, 3, 3], ['4', '2', '3']),
    ([42, 47], [2, 2], ['2', '3']),
    ([43, 45, 48], [1, 1, 1], ['2', '3', '4']),
    ([32, 35, 37], [0, 0, 0], ['3', '2', '4']),
]

####### Alternative points ########

red_data = [
    ([35, 45], [4, 4], ['0', '1']),
    ([15, 18, 21], [3, 3, 3], ['0', '1', '2']),
    ([38, 42], [2, 2], ['1', '0']),
    ([40, 48], [1, 1], ['1', '0']),
    ([45], [0], ['0']),
]

# Green data points
green_data = [
    ([18], [4], ['3']),
    ([9], [2 ], ['3']),
    ([14], [1], [ '3']),
    ([15,19], [0, 0], ['3', '4']),
]


# Initialize the plot
fig, ax = plt.subplots(figsize=(14, 10))

# Plot the red lines with markers
for x, y, labels in red_data:
    ax.plot(x, y, 'o-r')
    for xi, yi, label in zip(x, y, labels):
        ax.text(xi, yi + 0.2, label, fontsize=18, ha='center', va='bottom')

# Plot the green lines with markers
for x, y, labels in green_data:
    ax.plot(x, y, 'o-g')
    for xi, yi, label in zip(x, y, labels):
        ax.text(xi, yi + 0.2, label, fontsize=18, ha='center', va='bottom')

# Set axis labels and increase tick font sizes for clarity
ax.set_xlabel('Distance from Origin', fontsize=22)
ax.set_ylabel('Game Index', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=20)  # Increased tick label size

# Set y-axis to display integers only and increase y-axis limit slightly
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_ylim(-1, 5)  # Adjust the y-axis limit to ensure labels do not touch the top border

# Set the x-axis range from 0 to 50
ax.set_xlim(0, 50)
ax.set_xticks(range(0, 51, 5))  # Ticks every 5 units

# Display the plot
plt.savefig('graph_final_coals.png')
print('DONE!')
