from curses.ascii import SO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from constraints import Constraint, Variable
from grammar import *
import random




class WeightedSumConstraint(Constraint):
    """Represents a constraint where sum of weighted variables equals a target value.
    Used for Kakurasu puzzles where variables are weighted by their position."""
    
    def __init__(self, variables, weights, target, **kwargs):
        """Initialize a weighted sum constraint.
        
        Args:
            variables: Set of variables in this constraint
            weights: Dictionary mapping each variable to its weight (position)
            target: Target sum for the weighted variables
        """
        super().__init__(variables, target, **kwargs)
        self.weights = weights
        self.weighted_sum_expr = Sum(*[v * Number(weights[v]) for v in self.variables])

    def __str__(self) -> str:
        parts = []
        for v in sorted(self.variables, key=lambda v: v.name):
            weight = self.weights[v]
            parts.append(f"{weight}*{v}")
        return f"({' + '.join(parts)}) = {self.target} (p={self.get_p_correct():.2f})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(('weighted_equality', frozenset(self.variables), 
                    frozenset((v, self.weights[v]) for v in self.variables), 
                    self.target))

    def __eq__(self, other):
        return (isinstance(other, WeightedSumConstraint) and
                self.variables == other.variables and
                self.weights == other.weights and
                self.target == other.target)

    def copy(self):
        return WeightedSumConstraint(self.variables.copy(), self.weights.copy(), self.target)
    
    def get_effective_target(self, partial_assignment = None) -> int:
        """Return the effective target considering already assigned variables"""
        assigned_sum = sum(self.weights[v] * v.value 
                          for v in self.get_assigned() 
                          if v.value is not None)
        effective_target = self.target - assigned_sum
        
        # Account for variables in partial assignment
        if partial_assignment is not None:
            unassigned = self.get_unassigned()
            for var in partial_assignment:
                if var in unassigned:
                    effective_target -= self.weights[var] * partial_assignment[var]
        
        return effective_target
    
    def size(self):
        """Calculate the number of possible solutions for this constraint."""
        if self.test_contradiction():
            return 0
            
        count = sum(1 for _ in self.possible_solutions())
        return count
        
    def initial_size(self):
        """Calculate the initial size before any assignments."""
        if hasattr(self, '_initial_size'):
            return self._initial_size
            
        self._initial_size = sum(1 for _ in self.possible_solutions())
        return self._initial_size

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        # If no variables are unassigned, check the sum
        if not self.get_unassigned():
            current_sum = sum(self.weights[v] * v.value for v in self.variables if v.value is not None)
            return current_sum != self.target
            
        # Calculate minimum and maximum possible sums
        unassigned = self.get_unassigned()
        current_sum = sum(self.weights[v] * v.value for v in self.get_assigned() if v.value is not None)
        
        # Max sum: current + all unassigned set to 1
        max_possible = current_sum + sum(self.weights[v] for v in unassigned)
        
        # Min sum: just current sum (all unassigned set to 0)
        min_possible = current_sum
        
        # Contradiction if target is outside possible range
        return self.target < min_possible or self.target > max_possible

    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution"""
        # Calculate current sum from assigned and partially assigned variables
        current_sum = 0
        for v in self.get_assigned():
            if v.value is not None:
                current_sum += self.weights[v] * v.value
                
        # Add from partial assignment
        unassigned = self.get_unassigned()
        partial_vars = set()
        for v in unassigned:
            if v in partial_assignment:
                current_sum += self.weights[v] * partial_assignment[v]
                partial_vars.add(v)
        
        # Calculate bounds for remaining unassigned variables
        remaining = unassigned - partial_vars
        max_remaining = sum(self.weights[v] for v in remaining)
        
        # Check if target is still achievable
        return current_sum <= self.target <= current_sum + max_remaining

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment"""
        # Calculate weighted sum
        total = sum(self.weights[v] * assignment.get(v, v.value) 
                   for v in self.variables)
        return 1 if total == self.target else 0

    def sample(self, partial_assignment=None, subset_vars=None):
        """Sample a random assignment that satisfies the constraint"""
        # Handle existing assignment
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())
        
        if not self.is_consistent(assignment):
            return None
        
        # If subset_vars is None, use all unassigned variables; else restrict strictly
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            subset_vars = set(subset_vars) & vars_to_assign

        # If subset_vars is empty, do not introduce new bindings; just return if consistent
        if len(subset_vars) == 0:
            return assignment.copy() if self.is_consistent(assignment) else None
        
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            return assignment.copy() if self.evaluate(assignment) else None
        
        # For simplicity, use rejection sampling
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate a random assignment
            trial = assignment.copy()
            for var in subset_vars:
                trial[var] = random.choice([0, 1])
                
            # Check if it satisfies the constraint
            remaining_vars = vars_to_assign - subset_vars
            if not remaining_vars:
                # This is a complete assignment for this constraint
                if self.evaluate(trial):
                    return trial
            else:
                # We need to check if trial could lead to a valid solution
                if self.is_consistent(trial):
                    return trial
                    
        # If we couldn't find a solution with rejection sampling, try all possibilities
        for solution in self.possible_solutions(assignment, subset_vars):
            return solution  # Return the first solution
            
        return None  # No solution found

    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint"""
        base_assignment = {} if partial_assignment is None else partial_assignment.copy()
        assignment = base_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())
        
        if not self.is_consistent(assignment):
            return
        
        # If subset_vars is None, use all unassigned variables; otherwise restrict strictly
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            subset_vars = set(subset_vars) & vars_to_assign

        # If subset_vars is empty, just pass-through the base assignment if consistent
        if len(subset_vars) == 0:
            if self.is_consistent(assignment):
                yield assignment.copy()
            return
        
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            if self.evaluate(assignment):
                yield assignment.copy()
            return
        
        # If we're only generating solutions for a subset, emit assignments ONLY for that subset
        if subset_vars and subset_vars != vars_to_assign:
            # We'll need to recursively generate all assignments to subset_vars
            # and check if they could be part of a valid solution
            
            # Recursive helper function to generate all binary combinations for subset variables
            def generate_assignments(vars_list, current_idx, current_assignment):
                if current_idx >= len(vars_list):
                    # Base case: we've assigned all variables in the subset
                    # Check if this assignment could lead to a valid solution
                    if self.is_consistent(current_assignment):
                        # Trim to emit only subset variables plus original base keys
                        keep_keys = set(vars_list) | set(base_assignment.keys())
                        trimmed = {k: current_assignment[k] for k in current_assignment if k in keep_keys}
                        # Debug: ensure we are not leaking extra variables beyond subset
                        if len(trimmed) > (len(vars_list) + len(base_assignment)):
                            print("[DEBUG kakurasu subset emit] leaked_keys:", set(trimmed.keys()) - keep_keys)
                        if len(trimmed) != (len(vars_list) + len(base_assignment)):
                            print("[DEBUG kakurasu subset emit sizes]", {
                                "subset_vars": len(vars_list),
                                "base_keys": len(base_assignment),
                                "emitted": len(trimmed),
                            })
                        yield trimmed
                    return
                
                # Try both 0 and 1 for the current variable
                var = vars_list[current_idx]
                for val in [0, 1]:
                    current_assignment[var] = val
                    yield from generate_assignments(vars_list, current_idx + 1, current_assignment)
                
                # Remove the variable from the assignment for backtracking
                del current_assignment[var]
            
            # Convert subset_vars to a list for indexed access
            subset_list = list(subset_vars)
            yield from generate_assignments(subset_list, 0, assignment.copy())
            
        else:
            # We're generating complete solutions
            # Handle the case of one variable separately for efficiency
            if len(vars_to_assign) == 1:
                var = next(iter(vars_to_assign))
                effective_target = self.get_effective_target(assignment)
                
                # Check if setting var to 1 would satisfy the constraint
                if effective_target == self.weights[var]:
                    solution = assignment.copy()
                    solution[var] = 1
                    yield solution
                
                # Check if setting var to 0 would satisfy the constraint
                elif effective_target == 0:
                    solution = assignment.copy()
                    solution[var] = 0
                    yield solution
                    
                return
            
            # For multiple variables, recursively generate all binary combinations
            # and check each one
            vars_list = list(vars_to_assign)
            
            def generate_full_assignments(idx, current_assignment, remaining_target):
                if idx >= len(vars_list):
                    # Base case: we've assigned all variables
                    if remaining_target == 0:
                        yield current_assignment.copy()
                    return
                
                var = vars_list[idx]
                weight = self.weights[var]
                
                # Try setting var to 0
                current_assignment[var] = 0
                yield from generate_full_assignments(idx + 1, current_assignment, remaining_target)
                
                # Try setting var to 1 if it doesn't exceed the target
                if weight <= remaining_target:
                    current_assignment[var] = 1
                    yield from generate_full_assignments(idx + 1, current_assignment, remaining_target - weight)
                
                # Remove the variable from the assignment for backtracking
                del current_assignment[var]
            
            effective_target = self.get_effective_target(assignment)
            yield from generate_full_assignments(0, assignment.copy(), effective_target)

