import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
# from grammar import *
from collections import defaultdict
from typing import List, Tuple
from scipy.special import gammaln
import random
from utils import *
from math import comb, ceil, floor
# import math
from functools import lru_cache



def get_total_relatedness(constraints, unassigned_only = True):
    if len(constraints) < 2:
        return None

    overlaps = []
    for i in range(len(constraints)):
        for j in range(i+1, len(constraints)):
            c1 = constraints[i]
            c2 = constraints[j]
            vars_in_c1 = c1.get_unassigned() if unassigned_only else c1.get_variables()
            vars_in_c2 = c2.get_unassigned() if unassigned_only else c2.get_variables()
            vars_in_c1 = vars_in_c1
            vars_in_c2 = vars_in_c2
            total = len(vars_in_c1 | vars_in_c2)
            overlap = len(vars_in_c1 & vars_in_c2)/total if total > 0 else 0
            overlaps.append(overlap)
    return np.mean(overlaps) if overlaps else 0

def get_density(constraints):
    n = len(constraints)
    if n < 2:
        return 0
    actual_edges, max_edges = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            c1 = constraints[i]
            c2 = constraints[j]
            vars_in_c1 = c1.get_unassigned()
            vars_in_c2 = c2.get_unassigned()

            if len(vars_in_c1 & vars_in_c2) > 0:
                actual_edges += 1
            max_edges += 1
    return actual_edges / max_edges if max_edges > 0 else 0


def print_assignments(assignments, max_print=5):
    if not assignments or not assignments[0]:
        return

    order = sorted(assignments[0].keys())

    # Get the maximum width for each column based on variable names
    col_widths = {v: max(len(v.name), 1) for v in order}

    print()
    # Create header with proper spacing
    header = "  ".join(f"{v.name:{col_widths[v]}}" for v in order)
    print(header)
    print("-" * len(header))

    # Sort assignments
    ordered_assignments = sorted(assignments, key=lambda a: np.sum([2**a[v] for v in order]))

    # Print assignments with aligned columns
    for a in ordered_assignments[:max_print]:
        print("  ".join(f"{a[v]:{col_widths[v]}}" for v in order))

    if len(ordered_assignments) > max_print:
        print(f"\n(+{len(ordered_assignments)-max_print} more...)")
    print()



def get_variable_counts(assignments):
    values = {}
    for assignment in assignments:
        for variable in assignment:
            if variable.is_assigned():
                continue

            elif variable not in values:
                values[variable] = {}
                for v in variable.domain:
                    values[variable][v] = 0

            if assignment[variable] not in values[variable]:
                values[variable][assignment[variable]] = 0
            values[variable][assignment[variable]] += 1

    return values




def get_variable_probabilities(assignments):
    counts = get_variable_counts(assignments)
    probabilities = {}
    for variable, counts in counts.items():
        total = sum(counts.values())
        if total == 0:
            probabilities[variable] = None
        else:
            probabilities[variable] = {val: count/total if count > 0 else 0 for val, count in counts.items()}
    return probabilities


def get_variable_entropy(counts):
    total = sum(counts.values())

    probs = {val: count/total for val, count in counts.items()
            if count > 0}

    entropy = -sum(p * np.log2(p) for p in probs.values())
    return entropy

def get_variable_entropies(assignments):
    """Calculate entropy for each variable based on its value distribution in assignments."""
    counts = get_variable_counts(assignments)
    entropies = {}

    for var in counts:
        total = sum(counts[var].values())
        if total == 0:  # Skip variables with no counts
            continue

        entropy = get_variable_entropy(counts[var])
        entropies[var] = entropy

    return entropies

def get_most_certain_assignment(assignments):
    """Get variable with lowest entropy and its most probable value."""
    if not assignments:
        return None, None

    counts = get_variable_counts(assignments)
    entropies = get_variable_entropies(assignments)

    if not entropies:  # No valid entropies found
        return None, None

    # Find minimum entropy
    min_entropy = min(entropies.values())

    # Get all variables with this entropy
    min_entropy_vars = [var for var, entropy in entropies.items()
                       if entropy == min_entropy]

    # Randomly choose one
    min_entropy_var = random.choice(min_entropy_vars)

    # Get most probable value for that variable
    value_counts = counts[min_entropy_var]
    most_probable_value = max(value_counts.items(), key=lambda x: x[1])[0]

    return min_entropy_var, most_probable_value




def get_binary_variable_entropies(assignments):
    variable_probabilities = get_variable_probabilities(assignments)
    entropies = {}
    for v in variable_probabilities:
        p = variable_probabilities[v]
        if p == 0 or p == 1:
            entropy = 0
        else:
            entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        entropies[v] = entropy
    return entropies



def expand_assignments(assignments, constraint):
    return [new_assignment for assignment in assignments for new_assignment in constraint.possible_solutions(assignment)]


def integrate_new_constraint(assignments, constraint, subset_vars = None, max_size=None, return_none_if_too_large=True):

    if assignments is None:
        return None

    elif not assignments:
        #possible_solutions = list(constraint.possible_solutions())
        solutions = []
        for new_solution in constraint.possible_solutions():
            if subset_vars:
                tmp = new_solution.copy()
                for v in tmp:
                    if v not in subset_vars:
                        new_solution.pop(v)
            solutions.append(new_solution.copy())
            if max_size and len(solutions) > max_size:
                if return_none_if_too_large:
                    return None
                else:
                    return solutions[:max_size]


        if len(solutions) == 0 or len(list(solutions[0].keys())) == 0:
            return []
        else:
            return solutions

    else:

        integrated_assignments = []
        for assignment in assignments:

            new_solutions = list(constraint.possible_solutions(partial_assignment=assignment))
            for new_assignment in new_solutions:
                if subset_vars:
                    tmp = new_assignment.copy()
                    for v in tmp:
                        if v not in subset_vars:
                            new_assignment.pop(v)

                integrated_assignments.append(new_assignment.copy())

                if max_size and len(integrated_assignments) > max_size:
                    if return_none_if_too_large:
                        return None
                    else:
                        return integrated_assignments[:max_size]
                    #return integrated_assignments[:max_size]

        if not integrated_assignments:
            return None

        return integrated_assignments



def integrate_constraints(constraints, subset_vars = None):
    assignments = []  # Start with None instead of []
    for constraint in constraints:
        assignments = integrate_new_constraint(assignments, constraint, subset_vars=subset_vars)
        if assignments is None:
            return None
    return assignments
    

def constraint_cull(assignments, constraint):
    if assignments is None:
        return None

    elif not assignments:
        return []

    new_assignments = []
    for assignment in assignments:
         if constraint.is_consistent(assignment):
            new_assignments.append(assignment.copy())

    return new_assignments


def cull_assignments(assignments, constraints):
    if assignments is None:
        return None

    elif not assignments:
        return []

    new_assignments = [a.copy() for a in assignments]
    
    for constraint in constraints:
        new_assignments = constraint_cull(new_assignments, constraint)
        if new_assignments is None:
            return None
        if len(new_assignments) == 0:
            return []
    return new_assignments

def integrate_constraints_and_forget(assignments, constraints, memory_capacity=np.inf):

    if assignments is None:
        return [], 0

    new_assignments = integrate_new_constraint(assignments, constraints)

    if new_assignments is None:
        return [], sum([v.get_initial_entropy() for v in assignments[0].keys()]) if assignments else 0

    if len(new_assignments) == 0:
        return [], 0


    initial_entropy = np.sum([variable.get_entropy() for variable in new_assignments[0].keys()])

    vars = list(new_assignments[0].keys())
    true_entropy = calculate_joint_entropy(new_assignments)


    new_assignments = apply_combinatorial_capacity_noise(new_assignments, memory_capacity)

    current_entropy = calculate_joint_entropy(new_assignments) if new_assignments else 0


    if len(new_assignments) == 0:
        information_loss = initial_entropy
        assignments = []
    else:
        information_loss = true_entropy - current_entropy
        assignments = [a.copy() for a in new_assignments]

    return assignments, information_loss




def remove_redundant(assignments):
    """
    Remove redundant assignments by checking for duplicate key-value pairs.

    Args:
        assignments: List of assignment dictionaries

    Returns:
        List of unique assignments
    """
    if not assignments:
        return []

    new_assignments = []
    seen = set()

    for assignment in assignments:
        # Create a hashable representation of the assignment using key-value pairs
        # Sort the items to ensure consistent hashing
        hashable_assignment = frozenset(assignment.items())

        if hashable_assignment not in seen:
            new_assignments.append(assignment)
            seen.add(hashable_assignment)

    return new_assignments

def calculate_joint_entropy(solutions: list[dict]) -> float:
    if not solutions:
        return 0.0

    # return np.log2(len(solutions))
    return _calculate_joint_entropy(len(solutions))

@lru_cache
def _calculate_joint_entropy(n):
    return np.log2(n)



def get_solved_variables(assignments):
    if not assignments:
        return {}

    solved_variables = assignments[0].copy()
    for assignment in assignments[1:]:
        for var, val in assignment.items():
            if solved_variables.get(var) != val:
                solved_variables.pop(var, None)

    return solved_variables


def get_best_guess(values, tol=1e-6):
    confidences = [max(p_value, 1-p_value) for p_value in values.values()]
    best_confidence = max(confidences)
    best_guesses = [variable for variable in values if
                    abs(max(values[variable], 1-values[variable]) - best_confidence) < tol]
    return random.choice(best_guesses), best_confidence


def remove_solved_variables(assignments):
    if len(assignments) == 0 or len(assignments[0]) == 0:
        return assignments

    vars = list(assignments[0].keys())
    new_assignments = []
    for assignment in assignments:
        new_assignment = {v : assignment[v] for v in vars if not v.is_assigned()}
        new_assignments.append(new_assignment)
    return new_assignments



def get_simplified_assignments(assignments):
    if len(assignments) == 0 or len(assignments[0]) == 0:
        return []

    variable_probabilities = get_variable_probabilities(assignments)

    best_guesses = []
    max_certainty = 0
    for v in variable_probabilities:
        # Find the most likely value(s) for this variable
        most_likely_items = []
        max_prob = 0

        for val, prob in variable_probabilities[v].items():
            if prob > max_prob:
                max_prob = prob
                most_likely_items = [(val, prob)]
            elif prob == max_prob:
                most_likely_items.append((val, prob))

        # Randomly choose a value if there are multiple with the same probability
        most_likely, certainty = random.choice(most_likely_items)

        if certainty > max_certainty:
            max_certainty = certainty
            best_guesses = [(v, most_likely)]
        elif certainty == max_certainty:
            best_guesses.append((v, most_likely))

    best_guess, best_value = random.choice(best_guesses)

    simplified_assignments = []
    for assignment in assignments:
        if assignment[best_guess] == best_value:
            simplified_assignments.append(assignment)
    return simplified_assignments




def calculate_information_update(solutions_depth_n, solutions_depth_n_plus_1):

    if not solutions_depth_n or not solutions_depth_n_plus_1:
        return 0.0

    vars_depth_n = set().union(*(sol.keys() for sol in solutions_depth_n))
    assignment_frequencies = defaultdict(int)
    total_solutions = len(solutions_depth_n_plus_1)

    for solution in solutions_depth_n_plus_1:
        assignment = frozenset((var, solution[var]) for var in vars_depth_n)
        assignment_frequencies[assignment] += 1

    entropy = 0.0
    for count in assignment_frequencies.values():
        prob = count / total_solutions
        entropy -= prob * np.log2(prob)


    initial_entropy = np.log2(len(solutions_depth_n))

    information_gain = initial_entropy - entropy
    return information_gain

def calculate_assignment_kl(solutions_depth_n, solutions_depth_n_plus_1):
    """
    Calculate KL divergence between assignment distributions at consecutive depths.
    """
    if not solutions_depth_n or not solutions_depth_n_plus_1:
        return 0.0

    # Get variables from depth n
    vars_depth_n = set().union(*(sol.keys() for sol in solutions_depth_n))

    # If no variables overlap with new solutions, return 0
    if not any(any(var in sol for var in vars_depth_n) for sol in solutions_depth_n_plus_1):
        return 0.0

    # Count frequencies at depth n
    n_frequencies = defaultdict(int)
    for solution in solutions_depth_n:
        assignment = frozenset((var, solution[var]) for var in vars_depth_n)
        n_frequencies[assignment] += 1

    # Count frequencies at depth n+1
    n_plus_1_frequencies = defaultdict(int)
    for solution in solutions_depth_n_plus_1:
        assignment = frozenset((var, solution[var]) for var in vars_depth_n)
        n_plus_1_frequencies[assignment] += 1
        if assignment not in n_frequencies:
            return float("inf")  # Distribution impossible under prior

    # Calculate probabilities and KL divergence
    n_total = len(solutions_depth_n)
    n_plus_1_total = len(solutions_depth_n_plus_1)

    kl_divergence = 0.0
    for assignment in n_frequencies:
        p_n = n_frequencies[assignment] / n_total
        p_n_plus_1 = n_plus_1_frequencies[assignment] / n_plus_1_total

        if p_n > 0 and p_n_plus_1 > 0:
            kl_divergence += p_n_plus_1 * np.log2(p_n_plus_1 / p_n)
        elif p_n == 0 and p_n_plus_1 > 0:
            return float("inf")
        else:
            continue

    return kl_divergence

def get_IG_subproblem(assignments):
    if len(assignments) == 0 or len(assignments[0]) == 0:
        return 0
    initial_entropy = 0
    for v in assignments[0]:
        if v.is_assigned():
            raise ValueError(f"Variable {v} is assigned!")
        initial_entropy += v.get_entropy()

    current_entropy = calculate_joint_entropy(assignments)
    return initial_entropy - current_entropy





@lru_cache
def log_comb(n, k):
    # Ensure both n and k are valid
    if not (n >= 0 and k >= 0 and n >= k):
        # Return a very large negative number for invalid cases
        # or could raise an exception if preferred
        return float('-inf')
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / np.log(2)

def get_complexity(assignments):
    if len(assignments) == 0 or len(assignments[0]) == 0:
        return 0

    # Get number of assignments and variables
    n = len(assignments[0])  # number of variables per assignment
    k = len(assignments)     # number of assignments

    # Calculate total possible assignments based on domain sizes
    variables = list(assignments[0].keys())
    total_assignments = np.prod([len(v.domain) for v in variables])

    # Calculate log2(C(total_assignments, k))
    return log_comb(total_assignments, k)



def calculate_variable_entropy(assignments, var):
    """Calculate entropy of a single variable"""
    if not assignments:
        return 0.0

    counts = defaultdict(int)
    n = len(assignments)

    for assignment in assignments:
        if var in assignment:
            counts[assignment[var]] += 1

    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * np.log2(p)

    return entropy




def compute_forgetting_probability(assignments, capacity_bits, tolerance=0.01, max_iters=50, min_step=1e-4):
    if not assignments or not assignments[0]:
        return 0.0

    variables = list(assignments[0].keys())
    n_total_assignments = np.prod([len(v.domain) for v in variables])
    n_assignments = len(assignments)

    left, right = 0.0, 1.0
    target_p = 1.0

    test_info = log_comb(n_total_assignments, n_assignments)
    if test_info < capacity_bits:
        return 0.0

    # if log_comb(n_total_assignments, 1) > capacity_bits:
    #     return 1.0

    # Add maximum iterations and minimum step size
    for _ in range(max_iters):
        p = (left + right) / 2
        #n_keep = np.ceil(p * n_assignments)
        n_keep = p * n_assignments

        test_info = log_comb(n_total_assignments, n_keep)
        if abs(test_info - capacity_bits) < tolerance:
            target_p = p
            break
        elif test_info > capacity_bits:
            right = p
        else:
            target_p = p  # Keep track of best valid solution
            left = p

        if (right - left) < min_step:
            break
    return 1-target_p

def apply_combinatorial_capacity_noise(assignments, capacity_bits, tolerance=0.01, max_iters=50, min_step=1e-4):

    if not assignments or not assignments[0]:
        return assignments

    current_info = get_complexity(assignments)
    if current_info <= capacity_bits:
        return assignments

    forgetting_probability = compute_forgetting_probability(assignments, capacity_bits, tolerance=tolerance, max_iters=max_iters, min_step=min_step)

    # Calculate how many assignments to keep, ensuring it's a valid integer >= 0
    k = max(0, min(len(assignments), int(floor((1-forgetting_probability) * len(assignments)))))

    if k == 0:
        return []

    noisy_assignments = random.sample(assignments, k=k)
    return noisy_assignments





def get_lost_assignments(assignments, information_loss):
    if len(assignments) == 0 and information_loss > 0:
         return 2 ** information_loss

    elif len(assignments) == 0:
        return 0

    variables = set(assignments[0].keys())

    initial_entropy = np.sum([variable.get_entropy() for variable in variables])
    initial_size = 2 ** initial_entropy

    information_gain = get_IG_subproblem(assignments)

    size_from_culling =  initial_size * (1/2**(information_gain - information_loss ))
    missing_size = (size_from_culling - len(assignments))
    return missing_size


def get_sizes_from_culling(assignments, information_loss):
    if len(assignments) == 0 and information_loss > 0:
         return 2 ** information_loss

    elif len(assignments) == 0:
        return 0

    variables = set(assignments[0].keys())

    initial_entropy = np.sum([variable.get_entropy() for variable in variables])
    initial_size = 2 ** initial_entropy

    information_gain = get_IG_subproblem(assignments)

    size_from_culling =  initial_size * (1/2**(information_gain - information_loss ))
    missing_size = (size_from_culling - len(assignments))
    number_culled = initial_size - size_from_culling
    return number_culled,missing_size

def get_certainty_threshold(assignments, information_loss, beta_IL=1.0):

    missing_size = get_lost_assignments(assignments, information_loss)
    total_size = len(assignments) + missing_size
    if total_size == 0:
        return 1.0
    else:
        return (1.0 - beta_IL * missing_size/total_size)



def get_expected_correct_in_subproblem(assignments, certainty):
    if len(assignments) == 0 or len(assignments[0]) == 0:
        return 0
    variables = set(assignments[0].keys())
    expected_correct = 0
    for v in variables:
        if v.is_assigned():
            raise ValueError(f"Variable {v} is assigned!")

        max_a_priori = v.get_max_a_priori()
        p_correct = certainty + (1-certainty) * max_a_priori
        expected_correct += p_correct
    return expected_correct



def get_expected_correct(variables, assigned_only=False):
    expected_correct = 0
    for v in variables:
        if (not assigned_only) or (assigned_only and v.is_assigned()):
            expected_correct += v.get_p_correct()
    return expected_correct


def get_total_expected_correct(assignments, variables, certainty):
    vars_in_assignments, all_vars = set(), set(variables)

    if len(assignments) > 0 and len(assignments[0]) > 0:
        vars_in_assignments = set(assignments[0].keys())

    remaining_vars = all_vars - vars_in_assignments

    total_expected_correct = get_expected_correct_in_subproblem(assignments, certainty)
    total_expected_correct += get_expected_correct(remaining_vars, assigned_only=False)

    return total_expected_correct


def assign_variables(assignments, information_loss, certainty):
    solved_variables = get_solved_variables(assignments)

    if not solved_variables:
        return {}, assignments, information_loss

    n_solved = len(solved_variables)
    info_loss_per_variable = information_loss / n_solved

    for v in solved_variables:
        v.assign(solved_variables[v], certainty=certainty)
    information_loss -= info_loss_per_variable * n_solved
    assignments = remove_solved_variables(assignments)
    return solved_variables, assignments, information_loss


if __name__ == "__main__":
    from itertools import product
    from grammar import Variable
    variables = [Variable(f"v{i}") for i in range(5)]
    assignments = [dict(zip(variables, assignment)) for assignment in product([0, 1], repeat=5)]
    assignments = random.sample(assignments, 20)

    print(get_complexity(assignments))


