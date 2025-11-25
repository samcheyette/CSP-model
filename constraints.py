from itertools import combinations, product
from collections import defaultdict
from re import A
from typing import Dict, Set, Iterator, Tuple, List, Any
from models.CSP_working_model.grammar import Expression, Variable, Sum, Number, GreaterThan, LessThan
from math import comb
# from utils.utils import *
from models.CSP_working_model.utils.assignment_utils import *

import random
from collections import defaultdict
import numpy as np
from math import comb
import copy
import time
import sys
import os

def get_overlapping_constraints(all_constraints, assignments=[], partial_path=[]):
    remaining_constraints = [c for c in all_constraints if c not in partial_path]

    if len(assignments) == 0 and len(partial_path) == 0:

        return [(c,1) for c in remaining_constraints if len(c.get_unassigned()) > 0]

    # Get variables from assignments and path
    assignment_vars = set().union(*[set(a.keys()) for a in assignments]) if assignments else set()
    path_vars = set().union(*[c.get_unassigned() for c in partial_path]) if partial_path else set()
    all_vars = assignment_vars | path_vars

    # Score remaining constraints by overlap
    overlap_scores = []
    for constraint in remaining_constraints:
        constraint_vars = constraint.get_unassigned()
        if not constraint_vars.isdisjoint(all_vars):
            shared = len(constraint_vars & all_vars)
            total = len(constraint_vars | all_vars)
            overlap_scores.append((constraint, shared/total))

    return sorted(overlap_scores, key=lambda x: x[1], reverse=True)







def sort_constraints_by_relatedness(constraints):
    """Sort constraints to minimize introduction of new variables at each step.

    At each step, chooses the constraint that adds the fewest new variables
    relative to all variables seen in previous constraints.
    """
    if not constraints:
        return []

    # Convert input to list to avoid modifying the original
    constraints = list(constraints)

    # Start with smallest constraint
    sorted_constraints = []
    seen_variables = set()

    while constraints:  # Continue until all constraints are used
        # Find constraint that introduces fewest new variables
        best_score = (float('inf'), float('inf'))  # (new_vars, total_size)
        best_next = None
        best_idx = None

        for i, c in enumerate(constraints):
            # Count new variables this constraint would add
            c_vars = c.get_variables()
            new_vars = len(c_vars - seen_variables)
            total_size = len(c_vars)
            score = (new_vars, total_size)

            if score < best_score:
                best_score = score
                best_next = c
                best_idx = i

        if best_next is None or best_idx is None:  # Should never happen as constraints is not empty
            raise ValueError("Failed to find next constraint")

        # Add the chosen constraint and update seen variables
        sorted_constraints.append(best_next)
        seen_variables.update(best_next.get_variables())
        constraints.pop(best_idx)  # Now best_idx is guaranteed to be an int

    return sorted_constraints


def find_constraint_cliques(constraints, min_size=2, max_size=None):
    if not constraints:
        return []

    # Create adjacency dictionary: constraint -> {neighbors}
    # Two constraints are neighbors if they share at least one variable
    adjacency = {}
    for i, c1 in enumerate(constraints):
        neighbors = set()
        c1_vars = c1.get_variables()

        for j, c2 in enumerate(constraints):
            if i != j:
                c2_vars = c2.get_variables()
                # Check if they share any variables
                if not c1_vars.isdisjoint(c2_vars):
                    neighbors.add(c2)

        adjacency[c1] = neighbors

    # Use Bron-Kerbosch algorithm to find all maximal cliques
    def bron_kerbosch(R, P, X, cliques):
        """Recursive Bron-Kerbosch algorithm for finding maximal cliques.

        Args:
            R: Current partial clique
            P: Prospective nodes that could be added to R
            X: Nodes already processed
            cliques: List to collect found cliques
        """
        if not P and not X:
            if len(R) >= min_size and (max_size is None or len(R) <= max_size):
                cliques.append(R.copy())
            return

        # Choose pivot to minimize branching
        pivot_candidates = P.union(X)
        if pivot_candidates:
            # Choose pivot that has most neighbors in P
            pivot = max(pivot_candidates, key=lambda v: len(P.intersection(adjacency[v])))
            consider = P - adjacency[pivot]
        else:
            consider = P.copy()

        for v in list(consider):
            bron_kerbosch(
                R.union({v}),
                P.intersection(adjacency[v]),
                X.intersection(adjacency[v]),
                cliques
            )
            P.remove(v)
            X.add(v)

    all_cliques = []
    bron_kerbosch(set(), set(constraints), set(), all_cliques)

    all_cliques.sort(key=len, reverse=True)

    return all_cliques



def enumerate_related_constraint_subsets(constraints, min_length=1, max_length=None):
    """Lazily enumerate all subsets of constraints where each constraint shares variables with at least one other.

    A constraint is considered related to another if they share at least one variable.
    For subsets of size 1, all constraints are considered valid.

    Args:
        constraints: List or set of constraints to analyze
        min_length: Minimum size of subsets to return (default: 1)
        max_length: Maximum size of subsets to return (default: no limit)

    Yields:
        Sets of constraints, ordered by size from min_length to max_length
    """
    if not constraints:
        return

    # Convert to list if not already
    constraints = list(constraints)

    # Set default max_length if not provided
    if max_length is None:
        max_length = len(constraints)

    # Build adjacency dictionary: constraint -> {related constraints}
    adjacency = {}
    for i, c1 in enumerate(constraints):
        related = set()
        c1_vars = c1.get_variables()

        for j, c2 in enumerate(constraints):
            if i != j:
                c2_vars = c2.get_variables()
                # Check if they share any variables
                if not c1_vars.isdisjoint(c2_vars):
                    related.add(c2)

        adjacency[c1] = related

    # Handle size-1 subsets if requested
    if min_length <= 1:
        for c in constraints:
            yield {c}

    # For sizes 2 and up, we need to find connected subsets
    # We'll use a breadth-first approach, building subsets incrementally by size

    # Start with all possible pairs of related constraints
    current_size = 2
    current_subsets = []

    for i, c1 in enumerate(constraints):
        for c2 in adjacency[c1]:
            if i < constraints.index(c2):  # Avoid duplicates
                current_subsets.append({c1, c2})

    # Yield subsets of size 2 if within requested range
    if min_length <= 2 <= max_length:
        for subset in current_subsets:
            yield subset

    # Incrementally build larger subsets
    while current_subsets and current_size < max_length:
        next_size = current_size + 1
        next_subsets = []

        # For each current subset, try to add one more related constraint
        for subset in current_subsets:
            # Find all constraints that are related to at least one constraint in the subset
            candidate_constraints = set()
            for c in subset:
                candidate_constraints.update(adjacency[c])

            # Remove constraints already in the subset
            candidate_constraints -= subset

            # Create new subsets by adding each candidate
            for candidate in candidate_constraints:
                new_subset = subset.copy()
                new_subset.add(candidate)

                # Check if this exact subset has already been generated
                if new_subset not in next_subsets:
                    next_subsets.append(new_subset)

        # Update for next iteration
        current_subsets = next_subsets
        current_size = next_size

        # Yield subsets if within requested range
        if min_length <= current_size <= max_length:
            for subset in current_subsets:
                yield subset



def get_constraint_list_neighbors(constraints):
    related_constraints = set()
    for constraint in constraints:
        related_constraints.update(constraint.get_neighbor_constraints())
    return list(related_constraints)


def get_new_constraints_of_variables(variables, constraints=[]):
    new_constraints = set()
    for variable in variables:
        for constraint in variable.get_active_constraints():
            if constraint not in constraints:
                new_constraints.add(constraint)
    return list(new_constraints)


def break_up_constraints(constraints, max_constraint_size: int = 4,
                        subset_size: int = 3, coverage_probability: float = 0.8,
                        random_seed = None):

    import random
    from itertools import combinations

    if random_seed is not None:
        random.seed(random_seed)

    result_constraints = []

    for constraint in constraints:
        constraint_vars = list(constraint.get_variables())

        if len(constraint_vars) <= max_constraint_size:
            # Small enough - keep original constraint
            result_constraints.append(constraint)
        else:
            # Large constraint - break into partial constraints

            # Generate all possible subsets of the specified size
            all_subsets = list(combinations(constraint_vars, min(subset_size, len(constraint_vars))))

            # Decide which subsets to include based on coverage_probability
            selected_subsets = []
            for subset in all_subsets:
                if random.random() < coverage_probability:
                    selected_subsets.append(subset)

            # Ensure we have at least some coverage if probability is too low
            if not selected_subsets and all_subsets:
                # If no subsets selected, pick a few randomly
                num_to_select = max(1, int(len(all_subsets) * 0.1))  # At least 10% coverage
                selected_subsets = random.sample(all_subsets, min(num_to_select, len(all_subsets)))

            # Create partial constraints for selected subsets
            for subset in selected_subsets:
                subset_vars = set(subset)

                # Copy constraint attributes if they exist
                kwargs = {}
                if hasattr(constraint, 'row') and constraint.row is not None:
                    kwargs['row'] = constraint.row
                if hasattr(constraint, 'col') and constraint.col is not None:
                    kwargs['col'] = constraint.col
                if hasattr(constraint, 'reset_first'):
                    kwargs['reset_first'] = constraint.reset_first
                if hasattr(constraint, 'guess'):
                    kwargs['guess'] = constraint.guess

                partial_constraint = PartialConstraint(constraint, subset_vars, **kwargs)
                result_constraints.append(partial_constraint)

    return result_constraints





def generate_random_constraints(n_variables,
                             n_constraints,
                             p_inequality,
                             avg_size,
                             sd_size, force_nontrivial=True,
                             solvable = True, attempts_remaining = 1000):


    if n_constraints is None:
        n_constraints = n_variables

    # Create pool of variables and assign random values
    variables = [Variable(f"v{i}") for i in range(n_variables)]
    n_ones = int(np.random.randint(1, max(1, n_variables-1)))
    assignments = [1 for _ in range(n_ones)] + [0 for _ in range(n_variables - n_ones)]
    random.shuffle(assignments)

    true_assignment = {var: assignments[i] for i, var in enumerate(variables)}
    constraints = []



    while len(constraints) < n_constraints:
        # Sample constraint size from normal distribution
        size = int(max(1, min(n_variables, round(np.random.normal(avg_size, sd_size)))))

        # Randomly select variables for this constraint
        constraint_vars = set(random.sample(variables, size))

        # Calculate sum of true values for selected variables
        true_sum = sum(true_assignment[var] for var in constraint_vars)


        # Generate constraint based on true values
        if np.random.random() < p_inequality:
            # For inequality, pick any target except the true sum
            possible_targets = list(range(size + 1))
            possible_targets.remove(true_sum)
            target = random.choice(possible_targets)

            # Determine if it should be greater_than based on relationship to true_sum
            greater_than = true_sum > target

            constraint = InequalityConstraint(constraint_vars, target, greater_than=greater_than)
        else:
            # For equality, use the actual sum as target
            constraint = EqualityConstraint(constraint_vars, true_sum)

        if force_nontrivial and constraint.size() <= 1:
            continue
        constraints.append(constraint)

    constraints = list(set(constraints))


    assignments = []
    for c in constraints:
        assignments = integrate_new_constraint(assignments, c)
        if assignments is None:
            return generate_random_constraints(n_variables, n_constraints, p_inequality, avg_size, sd_size, force_nontrivial, solvable, attempts_remaining - 1)

    solved_variables = get_solved_variables(assignments)
    solved = True
    for v in variables:
        if v not in solved_variables:
            solved = False
            break
    if solvable and not solved and attempts_remaining > 0:
        return generate_random_constraints(n_variables, n_constraints, p_inequality, avg_size, sd_size, force_nontrivial, solvable, attempts_remaining - 1)

    return constraints, true_assignment



class Constraint:
    """Base class for all constraints"""
    def __init__(self, variables, target: int, **kwargs):
        # Convert variables to a set if it's not already
        self.variables = set(variables)
        self.target = target


        self.row, self.col = None, None
        if "row" in kwargs:
            self.row = kwargs["row"]
        if "col" in kwargs:
            self.col = kwargs["col"]
        self.reset_first = kwargs.get("reset_first", False)
        self.guess = kwargs.get("guess", False)

        for var in self.variables:
            var.add_constraint(self)

    def __del__(self):
        """Cleanup when constraint is destroyed"""
        for var in self.variables:
            var.remove_constraint(self)

    def get_variables(self) -> Set[Variable]:
        """Return set of variables in this constraint"""
        return self.variables

    def get_unassigned(self) -> Set[Variable]:
        """Return set of unassigned variables"""
        return {var for var in self.variables if var.value is None}

    def get_assigned(self) -> Set[Variable]:
        """Return set of assigned variables"""
        return {var for var in self.variables if var.value is not None}

    def is_active(self):
        """Check if constraint has any unassigned variables"""
        return len(self.get_unassigned()) > 0

    def get_neighbor_constraints(self, constraint_subset=None):
        """Get constraints that share variables with this constraint"""
        neighbors = set()
        for var in self.get_unassigned():
            relevant_constraints = (var.constraints if constraint_subset is None
                                  else [c for c in var.constraints if c in constraint_subset])
            for constraint in relevant_constraints:
                if (constraint != self and
                    not self.get_unassigned().isdisjoint(constraint.get_unassigned())):
                    neighbors.add(constraint)
        return neighbors

    def get_constraint_degree(self, constraint_subset=None) -> int:
        """Get number of other constraints this constraint shares variables with"""
        return len(self.get_neighbor_constraints(constraint_subset))

    def get_shared_variables(self, other: 'Constraint', unassigned_only = True) -> Set[Variable]:
        """Get variables shared between this constraint and another"""
        if unassigned_only:
            return self.get_unassigned() & other.get_unassigned()
        else:
            return self.get_variables() & other.get_variables()

    def get_variable_overlap(self, other: 'Constraint', unassigned_only = True) -> float:
        """Get fraction of variables shared with another constraint"""
        shared = len(self.get_shared_variables(other, unassigned_only))
        if unassigned_only:
            total = len(self.get_unassigned() | other.get_unassigned())
        else:
            total = len(self.get_variables() | other.get_variables())
        return shared / total if total > 0 else 0.0

    def get_effective_target(self, partial_assignment = None) -> int:
        """Return the effective target for this constraint"""
        effective_target = self.target - sum(var.value for var in self.get_assigned() if var.value is not None)
        unassigned = self.get_unassigned()
        if partial_assignment is not None:
            for key in partial_assignment:
                if key in unassigned:
                    effective_target -= partial_assignment[key]
        return effective_target


    def get_p_correct(self):
        p_correct = 1
        for v in self.get_assigned():
            p_correct *= v.get_p_correct()
        return p_correct

    def get_certainty(self):
        certainty = 1
        for v in self.get_assigned():
            certainty *= v.certainty
        return certainty

    def get_initial_entropy(self):
        return np.sum([v.get_initial_entropy() for v in self.get_variables()])

    def get_entropy(self):
        return np.log2(self.size()) if self.size() > 0 else self.get_initial_entropy()

    # def get_information_gain(self):
    #     return np.sum([v.get_information_gain() for v in self.get_assigned()])

    def get_information_gain(self):
        return self.get_initial_entropy() - self.get_entropy()

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment"""
        raise NotImplementedError()

    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution"""
        raise NotImplementedError()

    def sample(self, partial_assignment=None, subset_vars=None):
        """Sample a random assignment that satisfies the constraint"""
        raise NotImplementedError()

    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint"""
        raise NotImplementedError()

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()

    def initial_size(self):
        raise NotImplementedError()

    def copy(self):
        """Create a copy of this constraint"""
        raise NotImplementedError()

    def fix_contradiction(self):

        if not self.test_contradiction():
            return set()

        assigned_vars = list(self.get_assigned())
        if not assigned_vars:
            return set()  # No assigned variables to unassign

        # Remember original values to restore during testing
        original_values = {var: var.value for var in assigned_vars}

        # Try with increasingly larger subsets of variables
        for subset_size in range(1, len(assigned_vars) + 1):
            # Track which variables could potentially be causing the contradiction
            potential_offenders = set()

            # Try each combination of the current size
            for vars_to_unassign in combinations(assigned_vars, subset_size):
                # Temporarily unassign these variables
                for var in vars_to_unassign:
                    var.value = None

                # Check if contradiction is resolved
                if not self.test_contradiction():
                    # Add all these variables to the set of potential offenders
                    potential_offenders.update(vars_to_unassign)

                # Restore original values
                for var in vars_to_unassign:
                    var.value = original_values[var]

            # If we found any potential offenders at this size
            if potential_offenders:
                # Unassign all potential offending variables permanently
                for var in potential_offenders:
                    var.value = None

                return potential_offenders

        # If we reach here, unassigning all variables is the only solution
        for var in assigned_vars:
            var.unassign()

        return set(assigned_vars)

class EqualityConstraint(Constraint):
    """Represents a constraint where sum of variables equals a target value"""
    def __init__(self, variables: Set[Variable], target: int, **kwargs):
        super().__init__(variables, target, **kwargs)

        # Check if any variable has a non-binary domain
        has_non_binary = any(len(var.domain) > 2 or not var.domain.issubset({0, 1})
                           for var in variables)

        # Raise error for non-binary domains
        if has_non_binary:
            raise ValueError("EqualityConstraint only supports binary domains (0/1). "
                           "Use GeneralEqualityConstraint from special_constraints.generalized_constraints "
                           "for variables with non-binary domains.")

        self.sum_expr = Sum(*self.variables)

    def __str__(self) -> str:
        parts = []
        for v in self.variables:
            parts.append(str(v))  # Variable.__str__ will handle showing value if assigned
        return f"({' + '.join(parts)} = {self.target})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(('equality', frozenset(self.variables), self.target))

    def __eq__(self, other):
        return (isinstance(other, EqualityConstraint) and
                self.variables == other.variables and
                self.target == other.target)

    def copy(self):
        return EqualityConstraint(self.variables.copy(), self.target)

    def size(self):
        if self.test_contradiction():
            return 0
        else:
            effective_target = self.get_effective_target()
            n_unassigned = len(self.get_unassigned())
            return comb(n_unassigned, effective_target)

    def initial_size(self):
        return comb(len(self.variables), self.target)

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        effective_target = self.get_effective_target()
        n_unassigned = len(self.get_unassigned())

        if effective_target < 0 or effective_target > n_unassigned:
            return True
        return False

    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution"""
        initial_sum = 0
        for v in self.get_assigned():
            if v in partial_assignment and v.value != partial_assignment[v]:
                return 0

            initial_sum += v.value if v.value is not None else 0

        unassigned = self.get_unassigned()
        remaining = len(unassigned)
        for v in unassigned:
            if v in partial_assignment:
                initial_sum += partial_assignment[v]
                remaining -= 1

        return 1 if initial_sum <= self.target <= initial_sum + remaining else 0

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment"""
        effective_target = self.get_effective_target()
        return 1 if self.sum_expr.evaluate(assignment) == effective_target else 0

    def sample(self, partial_assignment=None, subset_vars=None):
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())

        if not self.is_consistent(assignment):
            return None

        # If subset_vars is None, use all unassigned variables
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            # Ensure subset_vars is a set and contains only variables from this constraint
            subset_vars = set(subset_vars)
            subset_vars = subset_vars & vars_to_assign

        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            return assignment.copy()

        # Calculate the target for all unassigned variables
        effective_target = self.get_effective_target(assignment)

        remaining_vars = vars_to_assign - subset_vars

        # Find possible targets for our subset
        subset_targets = []

        # If there are no variables outside our subset, we have just one target
        if not remaining_vars:
            subset_targets = [effective_target]
        else:
            # Calculate all possible sums the remaining variables could contribute
            # For each potential sum of remaining variables, calculate what our subset must sum to
            max_remaining_sum = min(len(remaining_vars), effective_target)

            for remaining_sum in range(max_remaining_sum + 1):
                subset_target = effective_target - remaining_sum
                # Only include valid targets (not negative, not larger than subset size)
                if 0 <= subset_target <= len(subset_vars):
                    subset_targets.append(subset_target)

        if len(subset_targets) == 0:
            return None

        # Choose a random target for our subset
        target = random.choice(subset_targets)

        # Choose a random valid assignment for this target
        # We need to select 'target' positions to place 1's out of len(subset_vars) positions
        subset_vars_list = sorted(subset_vars)

        # Calculate how many ways we can distribute 'target' 1's among subset_vars
        # This is the binomial coefficient (len(subset_vars) choose target)
        if len(subset_vars_list) < target:
            return None  # Safety check

        # Randomly select positions for the 1's
        ones_positions_indexes = sorted(random.sample(range(len(subset_vars_list)), target))
        ones_positions = [subset_vars_list[i] for i in ones_positions_indexes]

        # Create the solution
        solution = assignment.copy()
        for var in subset_vars:
            solution[var] = 1 if var in ones_positions else 0

        # If we have remaining variables outside our subset, we need to assign them too
        if remaining_vars:
            # We've made our subset sum to (effective_target - remaining_sum)
            # Now we need to make the remaining variables sum to remaining_sum
            remaining_sum = effective_target - target

            # Recursively sample a solution for the remaining variables with the new target
            if remaining_sum > 0:
                # Create a new constraint just for the remaining variables
                remaining_constraint = EqualityConstraint(remaining_vars, remaining_sum)
                remaining_solution = remaining_constraint.sample(solution)

                if remaining_solution is None:
                    return None  # No solution for remaining variables

                solution.update(remaining_solution)

        return solution

    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())

        if not self.is_consistent(assignment):
            return

        # If subset_vars is None, use all unassigned variables
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            # Ensure subset_vars is a set and contains only variables from this constraint
            subset_vars = set(subset_vars)
            subset_vars = subset_vars & vars_to_assign

        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            yield assignment.copy()
            return

        # Calculate the target for all unassigned variables
        effective_target = self.get_effective_target(assignment)

        remaining_vars = vars_to_assign - subset_vars

        # Find possible targets for our subset
        subset_targets = []

        # If there are no variables outside our subset, we have just one target
        if not remaining_vars:
            subset_targets = [effective_target]
        else:
            # Calculate all possible sums the remaining variables could contribute
            # The sum of remaining variables can be 0 up to min(remaining_size, effective_target)
            max_remaining_sum = min(len(remaining_vars), effective_target)

            # For each possible sum of remaining variables, calculate what our subset must sum to
            for remaining_sum in range(max_remaining_sum + 1):
                subset_target = effective_target - remaining_sum
                # Only include valid targets (not negative, not larger than subset size)
                if 0 <= subset_target <= len(subset_vars):
                    subset_targets.append(subset_target)

        if len(subset_targets) == 0:
            return

        # Generate solutions for each possible subset target
        for target in subset_targets:
            # Generate all ways to distribute 'target' 1's among subset_vars
            for ones_positions in combinations(sorted(subset_vars), target):
                solution = assignment.copy()
                for var in subset_vars:
                    solution[var] = 1 if var in ones_positions else 0
                yield solution


class InequalityConstraint(Constraint):
    """Represents sum(variables) < target or sum(variables) > target"""
    def __init__(self, variables, target, greater_than=False, **kwargs):
        super().__init__(variables, target, **kwargs)
        self.greater_than = greater_than
        self.sum_expr = Sum(*variables)

    def __str__(self):
        parts = []
        for v in self.variables:
            parts.append(str(v))
        symbol = ">" if self.greater_than else "<"
        return f"({' + '.join(parts)} {symbol} {self.target})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((frozenset(self.variables), self.target, self.greater_than))

    def __eq__(self, other):
        return (isinstance(other, InequalityConstraint) and
                self.variables == other.variables and
                self.target == other.target and
                self.greater_than == other.greater_than)

    def copy(self):
        return InequalityConstraint(self.variables.copy(), self.target, self.greater_than)

    def sample(self, partial_assignment=None, subset_vars=None):
        """Sample a random assignment that satisfies the constraint"""
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())

        if not self.is_consistent(assignment):
            return None

        # If subset_vars is None, use all unassigned variables
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            # Ensure subset_vars is a set and contains only variables from this constraint
            subset_vars = set(subset_vars)
            subset_vars = subset_vars & vars_to_assign

        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            return assignment.copy() if self.evaluate(assignment) else None

        # Calculate the current sum from assigned variables
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value

        # Calculate the remaining variables outside our subset
        remaining_vars = vars_to_assign - subset_vars

        if self.greater_than:
            # For ">" constraint: Need to ensure sum > target
            # Calculate the range of 1's we can place in our subset
            if not remaining_vars:
                # Need enough ones to make total > target
                min_ones = max(0, self.target + 1 - current_sum)
                if min_ones > len(subset_vars):
                    return None  # Impossible to satisfy
                max_ones = len(subset_vars)

                # Choose random number of 1's between min_ones and max_ones
                n_ones = random.randint(min_ones, max_ones)

                # Randomly select positions for the 1's
                subset_vars_list = sorted(subset_vars)
                ones_positions = set(random.sample(subset_vars_list, n_ones))

                # Assign values based on the selected positions
                for var in subset_vars:
                    assignment[var] = 1 if var in ones_positions else 0

                return assignment
            else:
                # We have remaining variables outside our subset
                # Choose a random number of 1's for our subset
                subset_ones = random.randint(0, len(subset_vars))
                subset_sum = subset_ones

                # Calculate how many 1's we need in the remaining vars
                min_remaining_ones = max(0, self.target + 1 - current_sum - subset_sum)

                # Check if it's possible to satisfy with remaining vars
                if min_remaining_ones > len(remaining_vars):
                    # Try again with a different number of ones in our subset
                    return self.sample(partial_assignment, subset_vars)

                # Assign values to our subset
                subset_vars_list = sorted(subset_vars)
                ones_positions = set(random.sample(subset_vars_list, subset_ones))

                for var in subset_vars:
                    assignment[var] = 1 if var in ones_positions else 0

                return assignment
        else:
            # For "<" constraint: Need to ensure sum < target
            if not remaining_vars:
                # Need few enough ones to keep total < target
                max_ones = min(self.target - current_sum - 1, len(subset_vars))

                if max_ones < 0:
                    return None  # Impossible to satisfy

                # Choose random number of 1's between 0 and max_ones
                n_ones = random.randint(0, max_ones)

                # Randomly select positions for the 1's
                subset_vars_list = sorted(subset_vars)
                ones_positions = set(random.sample(subset_vars_list, n_ones))

                # Assign values based on the selected positions
                for var in subset_vars:
                    assignment[var] = 1 if var in ones_positions else 0

                return assignment
            else:
                # We have remaining variables outside our subset
                # Choose a random number of 1's for our subset
                subset_ones = random.randint(0, len(subset_vars))
                subset_sum = subset_ones

                # Calculate the maximum 1's we can have in the remaining vars
                max_remaining_ones = self.target - current_sum - subset_sum - 1

                # Check if there's a possible assignment to remaining vars
                if max_remaining_ones < 0:
                    # Try again with a different number of ones in our subset
                    return self.sample(partial_assignment, subset_vars)

                # Assign values to our subset
                subset_vars_list = sorted(subset_vars)
                ones_positions = set(random.sample(subset_vars_list, subset_ones))

                for var in subset_vars:
                    assignment[var] = 1 if var in ones_positions else 0

                return assignment

    def initial_size(self):
        n_variables = len(self.variables)

        total_combinations = 1 << n_variables

        if self.greater_than:
            threshold = self.target + 1

            if threshold <= 0:
                return total_combinations
            elif threshold > n_variables:
                return 0
            else:
                valid_solutions = 0
                for k in range(threshold, n_variables + 1):
                    valid_solutions += comb(n_variables, k)
                return valid_solutions
        else:
            if self.target <= 0:
                return 0
            elif self.target > n_variables:
                return total_combinations
            else:
                valid_solutions = 0
                for k in range(self.target):
                    valid_solutions += comb(n_variables, k)
                return valid_solutions

    def size(self):
        if self.test_contradiction():
            return 0

        n_unassigned = len(self.get_unassigned())
        if n_unassigned == 0:
            # Only one solution (the current assignment)
            # Return 1 if it satisfies the constraint, 0 otherwise
            current_sum = sum(v.value if v.value is not None else 0 for v in self.variables)
            if self.greater_than:
                return 1 if current_sum > self.target else 0
            else:
                return 1 if current_sum < self.target else 0

        # Calculate current sum from assigned variables
        current_sum = sum(v.value if v.value is not None else 0 for v in self.get_assigned())

        # Total possible combinations for n_unassigned binary variables: 2^n_unassigned
        total_combinations = 1 << n_unassigned  # 2^n_unassigned

        if self.greater_than:
            # For sum > target, count solutions where sum is at least target+1
            threshold = self.target + 1 - current_sum

            if threshold <= 0:
                # Current sum already exceeds target, all solutions are valid
                return total_combinations
            elif threshold > n_unassigned:
                # Impossible to satisfy: even with all 1's we can't exceed target
                return 0
            else:
                # Count solutions with at least 'threshold' ones among n_unassigned variables
                # This is the sum of binomial coefficients: C(n,k) for k from threshold to n
                valid_solutions = 0
                for k in range(threshold, n_unassigned + 1):
                    valid_solutions += comb(n_unassigned, k)
                return valid_solutions
        else:
            # For sum < target, count solutions where sum is at most target-1
            threshold = self.target - current_sum

            if threshold <= 0:
                # Even with all 0's, we'd exceed or equal target
                return 0
            elif threshold > n_unassigned:
                # All combinations are valid since even with all 1's we'd be under target
                return total_combinations
            else:
                # Count solutions with at most 'threshold-1' ones among n_unassigned variables
                # This is the sum of binomial coefficients: C(n,k) for k from 0 to threshold-1
                valid_solutions = 0
                for k in range(threshold):
                    valid_solutions += comb(n_unassigned, k)
                return valid_solutions

    def evaluate(self, assignment):
        total = self.sum_expr.evaluate(assignment)
        effective_target = self.get_effective_target()
        return 1 if (total > effective_target if self.greater_than else total < effective_target) else 0

    def is_consistent(self, assignment):
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value

        remaining_vars = len(self.get_unassigned() - set(assignment.keys()))

        if self.greater_than:
            max_possible_sum = current_sum + remaining_vars
            return 1 if max_possible_sum > self.target else 0
        else:
            return 1 if current_sum < self.target else 0

    def test_contradiction(self):
        effective_target = self.get_effective_target()
        n_unassigned = len(self.get_unassigned())

        if self.greater_than:
            max_possible = n_unassigned
            return max_possible <= effective_target
        else:
            return effective_target <= 0

    def possible_solutions(self, partial_assignment=None, subset_vars=None):

        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())

        if not self.is_consistent(assignment):
            return

        # If subset_vars is None, use all unassigned variables
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            # Ensure subset_vars is a set and contains only variables from this constraint
            subset_vars = set(subset_vars) & vars_to_assign

        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            if self.evaluate(assignment):
                yield assignment.copy()
            return

        # Calculate the current sum from assigned variables
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value

        # Calculate the remaining variables outside our subset
        remaining_vars = vars_to_assign - subset_vars

        if self.greater_than:
            # For ">" constraint: Need to ensure sum > target
            # If no remaining variables outside our subset
            if not remaining_vars:
                # Need enough ones to make total > target
                remaining_needed = max(0, self.target + 1 - current_sum)
                if remaining_needed > len(subset_vars):
                    return  # Impossible to satisfy
                min_ones = remaining_needed
                max_ones = len(subset_vars)

                # Generate all possible assignments that make sum > target
                for n_ones in range(min_ones, max_ones + 1):
                    for ones_positions in combinations(sorted(subset_vars), n_ones):
                        solution = assignment.copy()
                        for var in subset_vars:
                            solution[var] = 1 if var in ones_positions else 0
                        yield solution
            else:
                # We have remaining variables outside our subset
                # For each possible number of ones in our subset, determine how many
                # ones we need in the remaining vars
                for subset_ones in range(len(subset_vars) + 1):
                    subset_sum = subset_ones
                    min_remaining_ones = max(0, self.target + 1 - current_sum - subset_sum)

                    # Only continue if it's possible to satisfy with remaining vars
                    if min_remaining_ones <= len(remaining_vars):
                        # Generate current subset assignment
                        for ones_positions in combinations(sorted(subset_vars), subset_ones):
                            subset_solution = assignment.copy()
                            for var in subset_vars:
                                subset_solution[var] = 1 if var in ones_positions else 0

                            # If we only want a solution for the subset_vars
                            # Check if this assignment to subset_vars can be part of a valid solution
                            temp_assignment = subset_solution.copy()
                            can_satisfy = False

                            # Check if we can satisfy the constraint with some assignment to remaining vars
                            if min_remaining_ones > 0:
                                # Need at least min_remaining_ones 1's in remaining_vars
                                if min_remaining_ones <= len(remaining_vars):
                                    can_satisfy = True
                            else:
                                # We already satisfy the constraint with just our subset
                                if current_sum + subset_sum > self.target:
                                    can_satisfy = True

                            if can_satisfy:
                                yield subset_solution
        else:
            # For "<" constraint: Need to ensure sum < target
            # If no remaining variables outside our subset
            if not remaining_vars:
                # Need few enough ones to keep total < target
                max_ones = min(self.target - current_sum - 1, len(subset_vars))
                min_ones = 0

                if max_ones < 0:
                    return  # Impossible to satisfy

                # Generate all possible assignments that make sum < target
                for n_ones in range(min_ones, max_ones + 1):
                    for ones_positions in combinations(sorted(subset_vars), n_ones):
                        solution = assignment.copy()
                        for var in subset_vars:
                            solution[var] = 1 if var in ones_positions else 0
                        yield solution
            else:
                # We have remaining variables outside our subset
                # For each possible number of ones in our subset
                for subset_ones in range(len(subset_vars) + 1):
                    subset_sum = subset_ones
                    max_remaining_ones = self.target - current_sum - subset_sum - 1

                    # Only continue if there's a possible assignment to remaining vars
                    if max_remaining_ones >= 0:
                        # Generate current subset assignment
                        for ones_positions in combinations(sorted(subset_vars), subset_ones):
                            subset_solution = assignment.copy()
                            for var in subset_vars:
                                subset_solution[var] = 1 if var in ones_positions else 0

                            # For subset_vars only, we just need to check if this assignment
                            # could be part of a valid complete solution
                            yield subset_solution


class PartialConstraint:
    """A wrapper for constraints that operates on a specific subset of variables.

    This allows a constraint to be treated as if it only contains the specified subset of variables,
    while maintaining access to the original constraint's behavior.
    """

    def __init__(self, constraint: Constraint, subset_vars: Set[Variable], **kwargs):
        """Initialize the wrapper with a constraint and subset of variables.

        Args:
            constraint: The original constraint
            subset_vars: The subset of variables to focus on (must be a subset of constraint's variables)
        """

        self.constraint = constraint
        self.variables = constraint.get_variables()
        # Ensure subset_vars is actually a subset of the constraint's variables
        self.subset_vars = set(subset_vars) & constraint.get_unassigned()
        self.reset_first = kwargs.get("reset_first", False)
        self.guess = kwargs.get("guess", False)
        self.row = kwargs.get("row", None)
        self.col = kwargs.get("col", None)
        if not self.row and not self.col:
            if hasattr(self.constraint, "row") and hasattr(self.constraint, "col"):
                self.row = self.constraint.row
                self.col = self.constraint.col


    def __str__(self) -> str:
        """Return string representation showing it's a subset of the original constraint."""
        return f"({self.constraint}, vars={{{','.join(str(v.name) for v in sorted(self.subset_vars, key=str))}}})"

    def __repr__(self) -> str:
        return self.__str__()

    def get_variables(self) -> Set[Variable]:
        """Return only the variables in the subset."""
        return self.subset_vars

    def get_unassigned(self) -> Set[Variable]:
        """Return unassigned variables from the subset."""
        return {v for v in self.subset_vars if v.value is None}

    def get_assigned(self) -> Set[Variable]:
        """Return assigned variables from the subset."""
        return {v for v in self.subset_vars if v.value is not None}

    def is_active(self):
        """Check if subset has any unassigned variables."""
        return len(self.get_unassigned()) > 0

    def get_constraint_degree(self, constraint_subset=None) -> int:
        """Pass through to original constraint's get_constraint_degree."""
        return self.constraint.get_constraint_degree(constraint_subset)

    def size(self):
        """Calculate size based on subset of variables."""
        # Use the original constraint's possible_solutions with our subset
        count = sum(1 for _ in self.possible_solutions())
        return count

    def evaluate(self, assignment):
        """Evaluate if the assignment satisfies the constraint for the subset."""
        return self.constraint.evaluate(assignment)

    def is_consistent(self, partial_assignment):
        """Check if partial assignment could lead to a valid solution."""
        return self.constraint.is_consistent(partial_assignment)

    def sample(self, partial_assignment=None):
        """Sample a random assignment for the subset."""
        return self.constraint.sample(partial_assignment, self.subset_vars)

    def possible_solutions(self, partial_assignment=None):
        """Generate all possible solutions for the subset."""
        yield from self.constraint.possible_solutions(partial_assignment=partial_assignment, subset_vars=self.subset_vars)

    def test_contradiction(self):
        """Pass through to original constraint's test_contradiction."""
        return self.constraint.test_contradiction()

    def fix_contradiction(self):
        """Fix the contradiction and return the unsolved variables."""
        return self.constraint.fix_contradiction()

    def get_certainty(self):
        return self.constraint.get_certainty()

    def get_p_correct(self):
        return self.constraint.get_p_correct()

    def copy(self):
        """Create a copy of this wrapper with copies of the constraint and subset."""
        return PartialConstraint(self.constraint.copy(), self.subset_vars.copy())

    def __eq__(self, other):
        return ((isinstance(other, PartialConstraint) and
                    (self.constraint == other.constraint)) and
                    (self.subset_vars == other.subset_vars))



class GiveUpConstraint(Constraint):
    def __init__(self, variables, **kwargs):
        super().__init__(variables, 0)
        self.row = kwargs.get("row", None)
        self.col = kwargs.get("col", None)


    def possible_solutions(self, partial_assignment=None):
        solution = {}
        for v in self.variables:
            if partial_assignment and v in partial_assignment:
                solution[v] = partial_assignment[v]
            elif v.value is None:
                solution[v] = int(v.sample())

        yield solution

    def __copy__(self):
        return GiveUpConstraint(self.variables.copy())


    def __str__(self):
        return f"GiveUp!"

    def test_contradiction(self):
        return False


def get_overlap_score(constraint, constraints):
    constraint_vars = constraint.get_unassigned()
    scores = []
    for i, c in enumerate(constraints):
        c_vars = c.get_unassigned()
        shared = len(constraint_vars & c_vars)
        total = len(constraint_vars | c_vars)
        if total > 0:
            scores.append(shared/total)

    return np.mean(scores) if scores else 0


def get_new_vars_added(constraint, vars):
    return len(constraint.get_unassigned() - vars)


def sample_related_constraint(all_constraints, constraints_in_path=[], n_samples=25, tau=0.25, max_vars=3):
    vars_in_path = set().union(*[c.variables for c in constraints_in_path])
    if not constraints_in_path:
        sampled_constraints = random.choices(all_constraints,k=n_samples)
        min_size = min([c.size() for c in sampled_constraints])
        sampled_constraints = [c for c in sampled_constraints if c.size() == min_size]

        constraint = random.choice(sampled_constraints)

        n_vars = min(max_vars, len(constraint.get_unassigned()))
        vars = random.sample(list(constraint.get_unassigned()), k=n_vars)
        return PartialConstraint(constraint, set(vars))
    else:
        sampled_constraints = []
        scores = []
        list_vars = list(vars_in_path)
        for _ in range(n_samples):
            var = random.choice(list_vars)
            neighbors = list(var.constraints)
            if neighbors:
                constraint = random.choice(neighbors)

                #max_vars = min(max_vars, len(constraint.get_unassigned()))
                #n_vars = np.random.randint(0, max_vars) if max_vars > 0 else 0
                n_vars = min(max_vars, len(constraint.get_unassigned()))

                vars = random.sample(list(constraint.get_unassigned()), k=n_vars)
                sample = PartialConstraint(constraint, set(vars))
                if sample not in constraints_in_path and sample not in sampled_constraints:

                    #score = get_overlap_score(sample, constraints_in_path)
                    score = -get_new_vars_added(sample, vars_in_path) + get_overlap_score(sample, constraints_in_path)
                    sampled_constraints.append(sample)
                    scores.append(score)


        if not sampled_constraints:
            return sample_related_constraint(all_constraints, constraints_in_path = [], n_samples=n_samples,
                                              tau=tau, max_vars=max_vars)
        else:

            ps = softmax(scores, tau)
            choice = sampled_constraints[np.random.choice(len(sampled_constraints), p=ps)]

            return choice









if __name__ == "__main__":
    v0 = Variable("v0", domain = {0,1})
    v1 = Variable("v1", domain = {0,1})
    v2 = Variable("v2", domain = {0,1})
    v3 = Variable("v3", domain = {0,1})

    c1 = EqualityConstraint({v1, v2}, 1)
    c2 = EqualityConstraint({v0,  v2}, 1)

    v1.assign(1)

    print(c1)






