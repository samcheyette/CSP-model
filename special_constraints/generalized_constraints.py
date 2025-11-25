from itertools import combinations, product
from collections import defaultdict
from typing import Dict, Set, Iterator, Tuple, List, Any
from math import comb
import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar import Expression, Variable, Sum, Number
from constraints import Constraint


class GeneralEqualityConstraint(Constraint):
    """
    A generalized version of EqualityConstraint that handles variables with arbitrary domains.
    This constraint enforces that the sum of the variables equals the target value.
    """
    def __init__(self, variables: Set[Variable], target: int, **kwargs):
        super().__init__(variables, target, **kwargs)
        self.sum_expr = Sum(*self.variables)
        
        # Store domains for fast access
        self.domains = {var: var.domain for var in variables}
        
        # Check if any variable has a non-binary domain
        self.has_non_binary = any(len(domain) > 2 or not domain.issubset({0, 1}) 
                                for domain in self.domains.values())

    def __str__(self) -> str:
        parts = []
        for v in self.variables:
            parts.append(str(v))  # Variable.__str__ will handle showing value if assigned
        return f"({' + '.join(parts)}) = {self.target} (p={self.get_p_correct():.2f})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(('general_equality', frozenset(self.variables), self.target))

    def __eq__(self, other):
        return (isinstance(other, GeneralEqualityConstraint) and
                self.variables == other.variables and
                self.target == other.target)

    def copy(self):
        return GeneralEqualityConstraint(self.variables.copy(), self.target)
    
    def size(self):
        """
        Calculate the number of possible solutions to this constraint using dynamic programming.
        For non-binary domains, we can compute this without enumerating all solutions.
        """
        if self.test_contradiction():
            return 0
            
        # Get unassigned variables
        unassigned = list(self.get_unassigned())
        
        if not unassigned:
            # If all variables are assigned, return 1 if constraint is satisfied, 0 otherwise
            current_sum = sum(var.value for var in self.variables if var.value is not None)
            return 1 if current_sum == self.target else 0
            
        # Calculate current sum from assigned variables
        current_sum = sum(var.value for var in self.get_assigned() if var.value is not None)
        
        # Calculate the remaining target for unassigned variables
        remaining_target = self.target - current_sum
        
        # If target is negative, it's impossible to satisfy
        if remaining_target < 0:
            return 0
        
        # Use dynamic programming to calculate the number of ways to make the sum
        # dp[i][s] = number of ways to make sum s using the first i variables
        n = len(unassigned)
        max_sum = remaining_target
        
        # Initialize dp table
        dp = [[0] * (max_sum + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # Base case: One way to make sum 0 with 0 variables
        
        # Fill dp table
        for i in range(1, n + 1):
            var = unassigned[i-1]
            for val in sorted(var.domain):
                for s in range(max_sum + 1):
                    if s >= val:
                        dp[i][s] += dp[i-1][s-val]
        
        # Return the number of ways to make the remaining target
        return dp[n][remaining_target]
    
    def initial_size(self):
        """
        Calculate the initial size of the constraint (number of possible assignments)
        using dynamic programming rather than enumerating solutions.
        """
        # Get all variables
        variables = list(self.variables)
        n = len(variables)
        
        # If there are no variables, return 0 or 1 based on target
        if n == 0:
            return 1 if self.target == 0 else 0
        
        # If target is negative, it's impossible to satisfy
        if self.target < 0:
            return 0
            
        # Use dynamic programming to calculate the number of ways to make the sum
        # dp[i][s] = number of ways to make sum s using the first i variables
        max_sum = self.target
        
        # Initialize dp table
        dp = [[0] * (max_sum + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # Base case: One way to make sum 0 with 0 variables
        
        # Fill dp table
        for i in range(1, n + 1):
            var = variables[i-1]
            for val in sorted(var.domain):
                for s in range(val, max_sum + 1):
                    dp[i][s] += dp[i-1][s-val]
        
        # Return the number of ways to make the target
        return dp[n][self.target]

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        # Calculate current sum from assigned variables
        current_sum = sum(var.value for var in self.get_assigned() if var.value is not None)
        
        # Get unassigned variables
        unassigned = self.get_unassigned()
        
        if not unassigned:
            # If all variables are assigned, check if sum equals target
            return current_sum != self.target
        
        # Calculate min and max possible sums
        min_possible = current_sum + sum(min(var.domain) for var in unassigned)
        max_possible = current_sum + sum(max(var.domain) for var in unassigned)
        
        # Contradiction if target is outside possible range
        return self.target < min_possible or self.target > max_possible

    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution"""
        # Calculate current sum from assigned variables
        current_sum = 0
        for v in self.get_assigned():
            if v in partial_assignment and v.value != partial_assignment[v]:
                return 0
            current_sum += v.value if v.value is not None else 0
        
        # Add values from partial_assignment for unassigned variables
        unassigned = self.get_unassigned()
        for v in unassigned:
            if v in partial_assignment:
                current_sum += partial_assignment[v]
        
        # Get variables that are unassigned and not in partial_assignment
        remaining_vars = [v for v in unassigned if v not in partial_assignment]
        
        if not remaining_vars:
            # If all variables are assigned or in partial_assignment, check if sum equals target
            return 1 if current_sum == self.target else 0
        
        # Calculate min and max possible sums
        min_possible = current_sum + sum(min(var.domain) for var in remaining_vars)
        max_possible = current_sum + sum(max(var.domain) for var in remaining_vars)
        
        # Consistent if target is within possible range
        return 1 if min_possible <= self.target <= max_possible else 0

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment"""
        total = self.sum_expr.evaluate(assignment)
        return 1 if total == self.target else 0

    def _recursive_generate_solutions(self, vars_list, target, index=0, current_assignment=None, 
                                     current_sum=0):
        """
        Recursively generate all possible solutions that satisfy the constraint.
        
        Args:
            vars_list: List of variables to assign
            target: The target sum
            index: Current index in vars_list
            current_assignment: Current partial assignment
            current_sum: Current sum of values
            
        Yields:
            Dictionary mapping variables to values that satisfy the constraint
        """
        if current_assignment is None:
            current_assignment = {}
        
        # If we've assigned all variables, check if we've hit the target
        if index == len(vars_list):
            if current_sum == target:
                yield current_assignment.copy()
            return
        
        current_var = vars_list[index]
        
        # Skip variables that already have a value assigned
        if current_var.value is not None:
            new_sum = current_sum + current_var.value
            current_assignment[current_var] = current_var.value
            yield from self._recursive_generate_solutions(
                vars_list, target, index + 1, current_assignment, new_sum)
            del current_assignment[current_var]
            return
        
        # Try each possible value in the variable's domain
        for value in current_var.domain:
            # Prune if current sum + value > target and all remaining variables can't be 0
            new_sum = current_sum + value
            
            if new_sum > target:
                # Check if all remaining variables can have minimum values
                min_remaining = sum(min(var.domain) for var in vars_list[index+1:])
                if new_sum + min_remaining > target:
                    continue
            
            # Prune if current sum + value + max possible remaining < target
            max_remaining = sum(max(var.domain) for var in vars_list[index+1:])
            if new_sum + max_remaining < target:
                continue
                
            # This value might work, so let's try it
            current_assignment[current_var] = value
            yield from self._recursive_generate_solutions(
                vars_list, target, index + 1, current_assignment, new_sum)
        
        # Remove the variable from the assignment after we've tried all values
        if current_var in current_assignment:
            del current_assignment[current_var]
    
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
            subset_vars = set(subset_vars) & vars_to_assign
            
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            return assignment.copy()
        
        # Calculate the current sum from assigned variables and partial assignment
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value
        
        # Calculate the remaining target for unassigned variables
        remaining_target = self.target - current_sum
        
        # Generate all possible solutions for the subset
        solutions = list(self._recursive_generate_solutions(
            list(subset_vars), remaining_target, 0, {}, 0))
        
        if not solutions:
            return None
        
        # Choose a random solution
        solution = random.choice(solutions)
        result = assignment.copy()
        result.update(solution)
        
        return result
            
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint"""
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())
        
        if not self.is_consistent(assignment):
            return
        
        # If subset_vars is None, use all unassigned variables
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            subset_vars = set(subset_vars) & vars_to_assign
        
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            yield assignment.copy()
            return
        
        # Calculate the current sum from assigned variables and partial assignment
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value
                
        # Calculate the remaining target for unassigned variables
        remaining_target = self.target - current_sum
        
        # Generate all possible solutions for the subset
        for solution in self._recursive_generate_solutions(
                list(subset_vars), remaining_target, 0, {}, 0):
            result = assignment.copy()
            result.update(solution)
            yield result 


class GeneralInequalityConstraint(Constraint):
    """
    A generalized version of InequalityConstraint that handles variables with arbitrary domains.
    This constraint enforces that the sum of the variables is greater than or less than the target value.
    """
    def __init__(self, variables, target, greater_than=False, **kwargs):
        super().__init__(variables, target, **kwargs)
        self.greater_than = greater_than
        self.sum_expr = Sum(*self.variables)
        
        # Store domains for fast access
        self.domains = {var: var.domain for var in variables}
        
        # Check if any variable has a non-binary domain
        self.has_non_binary = any(len(domain) > 2 or not domain.issubset({0, 1}) 
                                for domain in self.domains.values())

    def __str__(self) -> str:
        parts = []
        for v in self.variables:
            parts.append(str(v))  # Variable.__str__ will handle showing value if assigned
        symbol = ">" if self.greater_than else "<"
        return f"({' + '.join(parts)}) {symbol} {self.target} (p={self.get_p_correct():.2f})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(('general_inequality', frozenset(self.variables), self.target, self.greater_than))

    def __eq__(self, other):
        return (isinstance(other, GeneralInequalityConstraint) and
                self.variables == other.variables and
                self.target == other.target and
                self.greater_than == other.greater_than)

    def copy(self):
        return GeneralInequalityConstraint(self.variables.copy(), self.target, self.greater_than)
    
    def size(self):
        """
        Calculate the number of possible solutions to this constraint using dynamic programming.
        For non-binary domains, we can compute this without enumerating all solutions.
        """
        if self.test_contradiction():
            return 0
            
        # Get unassigned variables
        unassigned = list(self.get_unassigned())
        
        if not unassigned:
            # If all variables are assigned, check if constraint is satisfied
            current_sum = sum(var.value for var in self.variables if var.value is not None)
            if self.greater_than:
                return 1 if current_sum > self.target else 0
            else:
                return 1 if current_sum < self.target else 0
            
        # Calculate current sum from assigned variables
        current_sum = sum(var.value for var in self.get_assigned() if var.value is not None)
        
        # Use dynamic programming to calculate possible values and then count those that satisfy the constraint
        # dp[i][s] = number of ways to make sum s using the first i variables
        n = len(unassigned)
        
        # Calculate max possible sum to determine DP table size
        max_possible_sum = current_sum + sum(max(var.domain) for var in unassigned)
        
        # Initialize dp table (padding for handling < case)
        dp = [[0] * (max_possible_sum + 1) for _ in range(n + 1)]
        dp[0][current_sum] = 1  # Base case: One way to make current_sum with 0 variables
        
        # Fill dp table
        for i in range(1, n + 1):
            var = unassigned[i-1]
            for val in sorted(var.domain):
                for s in range(max_possible_sum + 1):
                    if s - val >= 0 and dp[i-1][s-val] > 0:
                        dp[i][s] += dp[i-1][s-val]
        
        # Count solutions based on inequality type
        total = 0
        if self.greater_than:
            # For sum > target, count all sums above target
            for s in range(self.target + 1, max_possible_sum + 1):
                total += dp[n][s]
        else:
            # For sum < target, count all sums below target
            for s in range(min(self.target, max_possible_sum + 1)):
                total += dp[n][s]
                
        return total
    
    def initial_size(self):
        """
        Calculate the initial size of the constraint (number of possible assignments)
        using dynamic programming rather than enumerating solutions.
        """
        # Get all variables
        variables = list(self.variables)
        n = len(variables)
        
        # If there are no variables, return based on inequality and target
        if n == 0:
            if self.greater_than:
                return 1 if 0 > self.target else 0
            else:
                return 1 if 0 < self.target else 0
            
        # Calculate max possible sum to determine DP table size
        max_possible_sum = sum(max(var.domain) for var in variables)
        
        # Initialize dp table with zeros
        dp = [[0] * (max_possible_sum + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # Base case: One way to make sum 0 with 0 variables
        
        # Fill dp table
        for i in range(1, n + 1):
            var = variables[i-1]
            for val in sorted(var.domain):
                for s in range(val, max_possible_sum + 1):
                    dp[i][s] += dp[i-1][s-val]
        
        # Count solutions based on inequality type
        total = 0
        if self.greater_than:
            # For sum > target, count all sums above target
            for s in range(self.target + 1, max_possible_sum + 1):
                total += dp[n][s]
        else:
            # For sum < target, count all sums below target
            for s in range(self.target):
                total += dp[n][s]
                
        return total

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        # Calculate current sum from assigned variables
        current_sum = sum(var.value for var in self.get_assigned() if var.value is not None)
        
        # Get unassigned variables
        unassigned = self.get_unassigned()
        
        if not unassigned:
            # If all variables are assigned, check if inequality is satisfied
            if self.greater_than:
                return current_sum <= self.target
            else:
                return current_sum >= self.target
        
        # Calculate min and max possible sums
        min_possible = current_sum + sum(min(var.domain) for var in unassigned)
        max_possible = current_sum + sum(max(var.domain) for var in unassigned)
        
        # Check for contradiction based on inequality type
        if self.greater_than:
            # For sum > target, contradiction if max_possible <= target
            return max_possible <= self.target
        else:
            # For sum < target, contradiction if min_possible >= target
            return min_possible >= self.target

    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution"""
        # Calculate current sum from assigned variables
        current_sum = 0
        for v in self.get_assigned():
            if v in partial_assignment and v.value != partial_assignment[v]:
                return 0
            current_sum += v.value if v.value is not None else 0
        
        # Add values from partial_assignment for unassigned variables
        unassigned = self.get_unassigned()
        for v in unassigned:
            if v in partial_assignment:
                current_sum += partial_assignment[v]
        
        # Get variables that are unassigned and not in partial_assignment
        remaining_vars = [v for v in unassigned if v not in partial_assignment]
        
        if not remaining_vars:
            # If all variables are assigned or in partial_assignment, check if inequality is satisfied
            if self.greater_than:
                return 1 if current_sum > self.target else 0
            else:
                return 1 if current_sum < self.target else 0
        
        # Calculate min and max possible sums
        min_possible = current_sum + sum(min(var.domain) for var in remaining_vars)
        max_possible = current_sum + sum(max(var.domain) for var in remaining_vars)
        
        # Check consistency based on inequality type
        if self.greater_than:
            # For sum > target, consistent if max_possible > target
            return 1 if max_possible > self.target else 0
        else:
            # For sum < target, consistent if min_possible < target
            return 1 if min_possible < self.target else 0

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment"""
        total = self.sum_expr.evaluate(assignment)
        if self.greater_than:
            return 1 if total > self.target else 0
        else:
            return 1 if total < self.target else 0

    def _recursive_generate_solutions(self, vars_list, index=0, current_assignment=None, 
                                     current_sum=0):
        """
        Recursively generate all possible solutions that satisfy the constraint.
        
        Args:
            vars_list: List of variables to assign
            index: Current index in vars_list
            current_assignment: Current partial assignment
            current_sum: Current sum of values
            
        Yields:
            Dictionary mapping variables to values that satisfy the constraint
        """
        if current_assignment is None:
            current_assignment = {}
        
        # If we've assigned all variables, check if we've satisfied the inequality
        if index == len(vars_list):
            if (self.greater_than and current_sum > self.target) or \
               (not self.greater_than and current_sum < self.target):
                yield current_assignment.copy()
            return
        
        current_var = vars_list[index]
        
        # Skip variables that already have a value assigned
        if current_var.value is not None:
            new_sum = current_sum + current_var.value
            current_assignment[current_var] = current_var.value
            yield from self._recursive_generate_solutions(
                vars_list, index + 1, current_assignment, new_sum)
            del current_assignment[current_var]
            return
        
        # Try each possible value in the variable's domain
        for value in current_var.domain:
            new_sum = current_sum + value
            
            # For > constraint
            if self.greater_than:
                # Early pruning: if current sum + max possible remaining <= target, skip
                max_remaining = sum(max(var.domain) for var in vars_list[index+1:])
                if new_sum + max_remaining <= self.target:
                    continue
            # For < constraint
            else:
                # Early pruning: if current sum + min possible remaining >= target, skip
                min_remaining = sum(min(var.domain) for var in vars_list[index+1:])
                if new_sum + min_remaining >= self.target:
                    continue
                
            # This value might work, so let's try it
            current_assignment[current_var] = value
            yield from self._recursive_generate_solutions(
                vars_list, index + 1, current_assignment, new_sum)
        
        # Remove the variable from the assignment after we've tried all values
        if current_var in current_assignment:
            del current_assignment[current_var]
    
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
            subset_vars = set(subset_vars) & vars_to_assign
            
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            return assignment.copy() if self.evaluate(assignment) else None
        
        # Calculate the current sum from assigned variables and partial assignment
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value
        
        # Generate all possible solutions for the subset
        solutions = list(self._recursive_generate_solutions(
            list(subset_vars), 0, {}, current_sum))
        
        if not solutions:
            return None
        
        # Choose a random solution
        solution = random.choice(solutions)
        result = assignment.copy()
        result.update(solution)
        
        return result
            
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint"""
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())
        
        if not self.is_consistent(assignment):
            return
        
        # If subset_vars is None, use all unassigned variables
        if subset_vars is None:
            subset_vars = vars_to_assign
        else:
            subset_vars = set(subset_vars) & vars_to_assign
        
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            if self.evaluate(assignment):
                yield assignment.copy()
            return
        
        # Calculate the current sum from assigned variables and partial assignment
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value
        
        # Generate all possible solutions for the subset
        for solution in self._recursive_generate_solutions(
                list(subset_vars), 0, {}, current_sum):
            result = assignment.copy()
            result.update(solution)
            yield result 