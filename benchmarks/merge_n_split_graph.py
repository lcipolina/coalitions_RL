import matplotlib.pyplot as plt
from scipy.special import binom

def bell_number(n):
    """Calculate Bell number using dynamic programming approach."""
    bell = [0] * (n + 1)
    bell[0] = 1
    for i in range(1, n + 1):
        bell[i] = 0
        for j in range(i):
            bell[i] += binom(i-1, j) * bell[j]
    return bell[n]

def compute_merge_and_split_costs(n_agents):
    """Compute the merge and split costs for a given range of agents."""
    merge_costs = []
    split_costs = []
    for n in n_agents:
        merge_cost = 2**n - n - 1
        split_cost = bell_number(n)
        merge_costs.append(merge_cost)
        split_costs.append(split_cost)
    total_costs = [m + s for m, s in zip(merge_costs, split_costs)]
    return total_costs

def plot_costs(n_agents, total_costs, constant_cost_value=1):
    """Plot the computational costs with specified parameters."""
    constant_costs = [constant_cost_value] * len(n_agents)

    plt.figure(figsize=(7, 6)) # to make it same as the costs graphs
    plt.plot(n_agents, total_costs, label='Total Cost (Merge and Split)', marker='o', color='blue')
    plt.plot(n_agents, constant_costs, label='DRL (ours) (O(1) Method)', marker='x', color='red')


     # Adjust font size here
    plt.xlabel('Number of Agents', fontsize=16)
    plt.ylabel('Computational Cost Per Agent', fontsize=16)

    plt.title('Computational Costs Between Merge and Split and Ours')
    plt.legend()
    plt.grid(True)
    plt.savefig('merge_n_split_comp_costs1.png')
    plt.show()

def main(n_agents):
    """Main entry point for calculating and plotting computational costs."""
    # Compute costs for the merge and split method
    total_costs = compute_merge_and_split_costs(n_agents)

    # Plot costs for comparison
    plot_costs(n_agents, total_costs)



# Run the main function
if __name__ == "__main__":
    # Customize the number of agents
    num_agents = 5
    n_agents = list(range(1, num_agents + 1))
    main(n_agents)
