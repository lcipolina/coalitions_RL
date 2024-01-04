import numpy as np
from collections import Counter

def characteristic_cost_function(distances, k=1, alpha=1):
    # Possible coalitions
    coalitions = [
        [0], [1], [2],  # {A}, {B}, {C}
        [0, 1], [0, 2], [1, 2],  # {A, B}, {A, C}, {B, C}
        [0, 1, 2]  # {A, B, C}
    ]

    # Initialize a Counter to keep track of each agent's preferred coalition
    preferred_coalitions = Counter()

    # Loop through each coalition and calculate its cost
    for coal in coalitions:
        coal_distances = distances[coal]
        if len(coal_distances) == 1:
            cost = coal_distances[0]
        else:
            cost = (k * np.var(coal_distances) / (alpha * len(coal)))

        # Update each agent's preferred coalition if this one is better
        for agent in coal:
            if agent not in preferred_coalitions or cost < preferred_coalitions[agent][1]:
                preferred_coalitions[agent] = (coal, cost)

    # Find the coalition that has the majority agreement
    coalition_votes = Counter([tuple(item[0]) for item in preferred_coalitions.values()])
    most_common_coalition = coalition_votes.most_common(1)[0][0]

    # Agents not in the most common coalition will be left alone
    remaining_agents = [agent for agent in range(len(distances)) if agent not in most_common_coalition]
    best_coalition_structure = [list(most_common_coalition)] + [[agent] for agent in remaining_agents]

    return best_coalition_structure

# Test the function
k = 20
alpha = 1
distances_list = [
    np.array([0.1, 0.2, 0.9]),
    np.array([0.5, 0.6, 0.7]),
    np.array([0.2, 0.8, 0.9])
]

for distances in distances_list:
    best_coalition_structure = characteristic_cost_function(distances, k, alpha)
    print(f'For distance list {distances}, the best coalition structure is {best_coalition_structure}')
