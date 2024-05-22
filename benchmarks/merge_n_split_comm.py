'''Merge and Split method as described in the paper by Saad'''

''' the characteristic_function now calculates the net utility for a coalition by subtracting the communication cost, which is the sum of the power required by each user to broadcast to its farthest coalition member. The total communication cost is also computed at the end, considering all coalitions.'''
'''the power required for each user in a coalition to broadcast to its farthest user represents a cost that reduces the total available power for actual data transmission. Hence, the utility (or benefit) of forming a coalition would be the increase in transmission capability (e.g., throughput, reliability) minus the power used for intra-coalition communication.
   This algo is meant to show the energy consumption, not the full implementation.
'''

# COSTS ARE IN WATTS


from itertools import combinations, chain


import numpy as np
import matplotlib.pyplot as plt

import numpy as np



class MergeAndSplitAlgorithm:
    def __init__(self, N, P_bar, nu_0, sigma_2, kappa, alpha):
        self.N = N
        self.P_bar = P_bar
        self.nu_0 = nu_0
        self.sigma_2 = sigma_2
        self.kappa = kappa
        self.alpha = alpha
        self._initialize_distances()

    def _initialize_distances(self):
        self.distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.distances[i][j] = abs(i - j) * 2  # Define your distance function here

    def calculate_variance_of_distances(self):
        if self.N <= 1:
            return 0  # Return 0 or a suitable default value for variance when there's only one agent
        distances_flat = self.distances[np.triu_indices(self.N, k=1)]
        if len(distances_flat) > 0:  # Check if there are enough distances to calculate variance
            variance = np.var(distances_flat)
        else:
            variance = 0  # Default value if not enough distances
        return variance

    def characteristic_function(self, coalition):
        communication_cost = 0
        for user in coalition:
            farthest_user = max((u for u in coalition if u != user),
                                key=lambda u: self.distances[user-1][u-1], default=user)
            if farthest_user == user:
                continue
            distance = self.distances[user-1][farthest_user-1]
            path_loss = (self.kappa / distance**self.alpha)
            communication_cost += (self.nu_0 * self.sigma_2) / path_loss
        return communication_cost

    def calculate_total_communication_cost(self):
        return self.characteristic_function(range(1, self.N + 1))




import matplotlib.pyplot as plt

def plot_combined_metrics_with_baseline(max_agents, P_bar, nu_0, sigma_2, kappa, alpha, conversion_factor=1000):
    num_agents = range(1, max_agents + 1)
    communication_costs = []
    distance_variances = []

    for N in num_agents:
        algorithm = MergeAndSplitAlgorithm(N, P_bar, nu_0, sigma_2, kappa, alpha)
        total_communication_cost = algorithm.calculate_total_communication_cost() / conversion_factor
        communication_costs.append(total_communication_cost)
        variance = algorithm.calculate_variance_of_distances()
        distance_variances.append(variance)

    plt.figure(figsize=(14, 6))

    # First Plot: Communication Cost vs. Number of Agents
    plt.subplot(1, 2, 1)
    plt.plot(num_agents, communication_costs, marker='o', linestyle='-', label='Merge and Split')
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2, label='DRL (ours)')
    plt.title('Communication Cost vs. Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Communication Cost (in mW)')
    plt.grid(True)
    plt.legend()

    # Second Plot: Communication Cost vs. Variance of Distances Among Agents
    plt.subplot(1, 2, 2)
    plt.scatter(distance_variances, communication_costs, c='blue', marker='o', label='Merge and Split')
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2, label='DRL (ours)')
    plt.title('Communication Cost vs. Variance of Distances Among Agents')
    plt.xlabel('Variance of Distances Among Agents')
    plt.ylabel('Communication Cost (in mW)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()




# Call the function with the parameters
P_bar = 100  # Total available power for coalition, in Watts
nu_0 = 1      # Target SNR for information exchange
sigma_2 = 1  # Noise variance, in Watts
kappa = 1.5  # Increased path loss constant
alpha = 3.5  # Increased path loss exponent

plot_combined_metrics_with_baseline(10, P_bar, nu_0, sigma_2, kappa, alpha, conversion_factor=1000)
