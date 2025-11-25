from curses.ascii import SO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from constraints import Constraint, Variable, PartialConstraint
import random



class UniquenessConstraint(Constraint):
    """Constraint that requires all variables to have different values, excluding given constants."""
    
    def __init__(self, variables, constants=None, **kwargs):
        """Initialize constraint.
        
        Args:
            variables: List of variables that must have unique values
            constants: List of values that are already used/fixed
        """
        # Call parent constructor with just the variables
        super().__init__(variables, target=0)  # Set a default target value
        
        # Store constants separately
        self.constants = set(constants) if constants is not None else set()
        if "row" in kwargs:
            self.row = kwargs["row"]
        if "col" in kwargs:
            self.col = kwargs["col"]
        
        # Verify all variables have same domain
        domains = {frozenset(v.domain) for v in variables}
        if len(domains) != 1:
            raise ValueError("All variables in UniquenessConstraint must have same domain")
        
        # Verify constants are valid
        if self.constants:
            domain = next(iter(domains))  # We know all domains are same
            invalid = self.constants - domain
            if invalid:
                raise ValueError(f"Constants {invalid} not in variable domain {domain}")
            

    def __eq__(self, other):
        return (isinstance(other, UniquenessConstraint) and
                self.variables == other.variables and
                self.constants == other.constants)
    
    def __hash__(self):
        return hash(('uniqueness', frozenset(self.variables), frozenset(self.constants)))
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        """Return string representation using ≠ notation."""
        vars_str = " ≠ ".join(str(v) for v in sorted(self.variables, key=str))
        if self.constants:
            # Add constants at the end
            vars_str += f" ≠ {{{','.join(map(str, sorted(self.constants)))}}}"
        return vars_str
    
    def size(self):
        """Calculate the number of possible solutions for this constraint.
        
        For a uniqueness constraint, the size is a falling factorial: 
        (domain_size - constants_size) * (domain_size - constants_size - 1) * ... * (domain_size - constants_size - n + 1)
        where n is the number of unassigned variables.
        
        If there's a contradiction, return 0.
        """
        if self.test_contradiction():
            return 0
            
        domain = next(iter(self.variables)).domain
        domain_size = len(domain)
        
        n_unassigned = len(self.get_unassigned())
        
        if n_unassigned == 0:
            return 1 
            
        # Calculate available values (domain minus used values and constants)
        used_values = {v.value for v in self.get_assigned()}
        available_values = domain_size - len(self.constants) - len(used_values - self.constants)
        
        # Check if we have enough values for remaining variables
        if available_values < n_unassigned:
            return 0  
            
        # Calculate falling factorial: available_values * (available_values-1) * ... * (available_values-n_unassigned+1)
        result = 1
        for i in range(n_unassigned):
            result *= (available_values - i)
            
        return result
    
        
    def copy(self):
        """Return a copy of this constraint"""
        return UniquenessConstraint(self.variables.copy(), constants=self.constants.copy())
    
    def evaluate(self, assignment):
        """Check if all assigned variables have different values and don't use constants."""
        values = [assignment[v] for v in self.variables if v in assignment]
        # Check no duplicates and no overlap with constants
        return (len(values) == len(set(values)) and 
                not (set(values) & self.constants))
    
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        
        # Get all unassigned variables
        unassigned = self.get_unassigned() - set(assignment.keys())
        
        # If subset_vars is None, use all unassigned variables
        if subset_vars is None:
            vars_to_assign = unassigned
        else:
            # FIXED: Ensure subset_vars only contains variables from this constraint
            subset_vars = set(subset_vars)
            vars_to_assign = subset_vars & unassigned & self.variables  # Added & self.variables
        
        # Variables outside the subset we need to consider
        remaining_vars = unassigned - vars_to_assign
        
        if not self.is_consistent(assignment):
            return
            
        # If no variables to assign, check if the constraint is satisfied
        if not unassigned:
            if self.evaluate(assignment):
                yield assignment.copy()
            return
        
        # If subset_vars is specified but no variables from this constraint are in the subset,
        # check if we should return the partial assignment or nothing
        if subset_vars is not None and not vars_to_assign:
            # If there are remaining variables in the constraint that need to be satisfied,
            # AND we have a non-empty partial assignment, check if the constraint can still be satisfied
            if remaining_vars and assignment:
                # Get currently used values from the constraint's variables
                used_values = {assignment[v] for v in self.variables if v in assignment}
                used_values.update(v.value for v in self.variables if v.value is not None)
                
                # Check if there are enough available values for the remaining variables
                domain = next(iter(self.variables)).domain
                available_values = domain - used_values - self.constants
                
                if len(available_values) >= len(remaining_vars):
                    # The constraint can be satisfied, return the partial assignment
                    yield assignment.copy()
            # If no remaining variables and we have a partial assignment, check if constraint is satisfied
            elif not remaining_vars and assignment:
                if self.evaluate(assignment):
                    yield assignment.copy()
            # If no partial assignment (empty assignment), return nothing
            # (This handles the case where subset_vars contains no constraint variables)
            return
        
        # If we're only generating solutions for a subset of variables
        if vars_to_assign and subset_vars is not None:
            # Need to check if assignments to vars_to_assign could be part of a valid solution
            
            # Get currently used values
            used_values = {assignment[v] for v in self.variables if v in assignment}
            used_values.update(v.value for v in self.variables if v.value is not None)  # Add assigned values
            
            # Get domain of first variable (all have same domain)
            domain = next(iter(self.variables)).domain
            
            # Get available values
            available_values = domain - used_values - self.constants
            
            # Check if there are enough values for all variables
            if len(available_values) < len(vars_to_assign):
                return  # Not enough values for subset
            
            # Generate all possible assignments to the subset
            # Recursive helper function to build assignments to subset
            def generate_subset_assignments(current_assignment, vars_left, available_vals):
                if not vars_left:
                    yield current_assignment.copy()
                    return
                
                var = next(iter(vars_left))
                new_vars_left = vars_left - {var}
                
                for val in available_vals:
                    new_assignment = current_assignment.copy()
                    new_assignment[var] = val
                    new_available = available_vals - {val}
                    
                    yield from generate_subset_assignments(new_assignment, new_vars_left, new_available)
            
            # Generate and yield all valid assignments to the subset
            for subset_assignment in generate_subset_assignments(assignment.copy(), vars_to_assign, available_values):
                # Check if there's enough remaining values for the remaining variables
                if subset_vars is None or not remaining_vars:
                    yield subset_assignment
                else:
                    # Check if solution is still possible with the remaining variables
                    curr_used_values = used_values | {subset_assignment[v] for v in vars_to_assign}
                    remaining_available = domain - curr_used_values - self.constants
                    
                    if len(remaining_available) >= len(remaining_vars):
                        yield subset_assignment
            
            return
        
        # Original algorithm for the case where we want a complete solution
        # Get available values (domain minus used values, constants, and assigned values)
        used_values = {assignment[v] for v in self.variables if v in assignment}
        used_values.update(v.value for v in self.variables if v.value is not None)
        
        first_var = next(iter(unassigned))
        domain = next(iter(self.variables)).domain  
        available_values = domain - used_values - self.constants
        
        for value in available_values:
            new_assignment = assignment.copy()
            new_assignment[first_var] = value
            yield from self.possible_solutions(new_assignment, subset_vars)

    def is_consistent(self, assignment):
        """Check if current partial assignment could lead to solution."""
        # Get values from both assignment and assigned variables
        values = [assignment[v] for v in self.variables if v in assignment]
        values.extend(v.value for v in self.variables if v.value is not None)
        
        return (len(values) == len(set(values)) and  # No duplicates
                not (set(values) & self.constants))   # No overlap with constants

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        # Check if any assigned variables have same value
        values = [v.value for v in self.get_assigned()]
        if len(values) != len(set(values)):
            return True
        # Check if any assigned values are in constants
        if set(values) & self.constants:
            return True
        # Check if we have enough values left for remaining variables
        remaining_vars = len(self.get_unassigned())
        if remaining_vars > 0:
            domain = next(iter(self.variables)).domain
            available_values = domain - set(values) - self.constants
            if len(available_values) < remaining_vars:
                return True
        return False
    

if __name__ == "__main__":
    from utils.assignment_utils import *
    v0 = Variable("v0", {1,2,3})
    v1 = Variable("v1", {1,2,3})
    v2 = Variable("v2", {1,2,3})
    v3 = Variable("v3", {1,2,3})
    v4 = Variable("v4", {1,2,3})


    c1 = UniquenessConstraint({v0, v1})
    c2 = UniquenessConstraint({v0, v2, v3})

    p1 = PartialConstraint(c1, {v0, v1, v2})
    p2 = PartialConstraint(c2, set())
    
    v2.assign(1)

    constraints = [p1, p2]

    assignments = []

    for constraint in constraints:
        assignments = integrate_new_constraint(assignments, constraint)
        print(constraint)
        print_assignments(assignments, 10)