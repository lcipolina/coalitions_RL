'''Merge and Split replication code.
   I only tested that indeed it gives the grand coalition for a superadditive function.
   For now, it acts as if a centralized agent is calculating all the merges and splits (or only one agent is doing the cals)

   Also missing: For example, the algorithm should be implemented in a decentralized way + with communication:
   - Each agent proposes a coalition to the other N-1 agents
   - If they reject, the agents keep proposing
   - Since each agent proposes coalitions to the other agents,  the communication complexity it's like a double 'for loop': O(nbr agents ^2)
'''

def calculate_utility(coalition):
    """ Utility function that models superadditive characteristics. """
    return sum(coalition)**2  # Sum of elements squared, to enforce superadditivity

def can_merge(utility_current, utility_new):
    """ Check if merging improves total utility. """
    return utility_new > utility_current

def get_all_subsets(coalitions):
    """ Generate all non-empty subsets of the given set of coalitions. """
    from itertools import chain, combinations
    return list(chain(*[combinations(coalitions, r) for r in range(1, len(coalitions)+1)]))

def merge_and_split(coalitions):
    """ Perform one iteration of merge and split operations. """
    import itertools

    # Start with the initial set of coalitions
    current_structures = [{i} for i in coalitions]
    best_structure = current_structures[:]
    best_utility = sum([calculate_utility(c) for c in current_structures])

    # Attempt to merge coalitions
    for n in range(2, len(current_structures) + 1):
        for subset in itertools.combinations(current_structures, n):
            merged = set().union(*subset)
            new_structure = [s for s in current_structures if s not in subset] + [merged]
            new_utility = sum([calculate_utility(c) for c in new_structure])
            if can_merge(best_utility, new_utility):
                best_utility = new_utility
                best_structure = new_structure

    # Attempt to split each coalition in the best structure found
    final_structure = best_structure[:]
    for coalition in best_structure:
        if len(coalition) > 1:
            subsets = get_all_subsets(coalition)
            for subset in subsets:
                if subset != coalition:
                    remaining = coalition.difference(subset)
                    new_structure = [c for c in final_structure if c != coalition] + [set(subset), remaining]
                    new_utility = sum([calculate_utility(c) for c in new_structure])
                    if new_utility > best_utility:
                        best_utility = new_utility
                        final_structure = new_structure

    return final_structure

# Example usage
agents = set(range(1, 11))  # Simulating 10 agents
final_coalitions = merge_and_split(agents)
print("Final coalitions:", final_coalitions)
