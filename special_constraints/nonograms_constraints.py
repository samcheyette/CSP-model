from curses.ascii import SO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from constraints import Constraint, Variable
import random
from math import comb




class RunLengthsConstraint(Constraint):
    """Represents a constraint for Nonogram run lengths in a row or column.
    
    This constraint ensures that a sequence of binary variables matches the specified
    run lengths of consecutive 1's, with at least one 0 between each run.
    """
    
    def __init__(self, variables, run_lengths, variable_order = None, **kwargs):
        """Initialize a run lengths constraint for Nonograms.
        
        Args:
            variables: Set of variables in this constraint
            run_lengths: List of integers specifying the lengths of consecutive runs of 1's
            variable_order: List of variables in the order they appear in the row/column.
                            If None, will use sorted(variables) which may not be correct for nonograms.
        """
        target = sum(run_lengths)
        super().__init__(variables, target, **kwargs)
        
        # Store run lengths and variable order
        self.run_lengths = [l for l in run_lengths if l > 0]
        self.variable_order = variable_order if variable_order is not None else sorted(list(variables), key=lambda v: v.name)
        
        # Validate that variable_order contains all variables
        if set(self.variable_order) != self.variables:
            raise ValueError("variable_order must contain exactly the same variables as the variables parameter")
        
        # Validate variables are binary
        for var in variables:
            if var.domain != {0, 1}:
                raise ValueError(f"All variables in RunLengthsConstraint must be binary, got {var.domain}")
        
        # Validate that run lengths can fit in the variables
        min_length_needed = sum(run_lengths) + len(run_lengths) - 1
        if min_length_needed > len(variables):
            raise ValueError(f"Run lengths {run_lengths} require at least {min_length_needed} variables, but only {len(variables)} provided")

    def __str__(self) -> str:
        """String representation showing the variables and run lengths."""
        var_str = " ".join(str(v) for v in self.variable_order)
        run_str = ",".join(str(length) for length in self.run_lengths)
        return f"[{var_str}] must have runs {run_str} (p={self.get_p_correct():.2f})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(('runlengths', frozenset(self.variables), tuple(self.run_lengths)))

    def __eq__(self, other):
        return (isinstance(other, RunLengthsConstraint) and
                self.variables == other.variables and
                self.run_lengths == other.run_lengths)

    def copy(self):
        """Create a copy of this constraint."""
        return RunLengthsConstraint(self.variables.copy(), list(self.run_lengths), list(self.variable_order))
    
    def size(self):
        """Calculate the number of possible solutions to this constraint."""
        if self.test_contradiction():
            return 0
        
        # Count the number of valid configurations for unassigned variables
        count = sum(1 for _ in self.possible_solutions())
        return count
    
    def initial_size(self):
        """Calculate the initial number of possible solutions before any assignments."""
        n = len(self.variables)
        k = len(self.run_lengths)
        
        # If there are no run lengths, the only valid solution is all 0s
        if k == 0:
            return 1
        
        # Calculate the minimum length needed: sum of run lengths plus at least one gap between runs
        min_length_needed = sum(self.run_lengths) + k - 1
        
        # If we don't have enough variables, return 0
        if min_length_needed > n:
            return 0
        
        # Number of ways to place k runs with specified lengths in a sequence of length n
        # This is equivalent to distributing the remaining "free" positions among k+1 positions
        # (before first run, between runs, after last run)
        return comb(n - sum(self.run_lengths) + 1, k)
    
    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy."""
        # Get the current partial assignment
        assignment = {v: v.value for v in self.get_assigned() if v.value is not None}
        
        # If there are no assigned variables, there's no contradiction
        if not assignment:
            return False
        
        # Check if any partial solution exists with the current assignments
        try:
            next(self._validate_partial_assignment(assignment))
            return False  # At least one solution exists
        except StopIteration:
            return True   # No solutions exist
    
    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution."""
        try:
            next(self._validate_partial_assignment(partial_assignment))
            return 1  # At least one solution exists
        except StopIteration:
            return 0  # No solutions exist
    
    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by an assignment."""
        # Create a binary sequence from the assignment
        sequence = []
        for v in self.variable_order:
            if v in assignment:
                sequence.append(assignment[v])
            elif v.value is not None:
                sequence.append(v.value)
            else:
                return 0  # Incomplete assignment
        
        # Count the runs of 1's in the sequence
        runs = []
        current_run = 0
        for bit in sequence:
            if bit == 1:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        
        # Add the last run if there is one
        if current_run > 0:
            runs.append(current_run)
        
        # Check if the runs match the expected run lengths
        return 1 if runs == self.run_lengths else 0
    
    def sample(self, partial_assignment=None, subset_vars=None):
        """Sample a random assignment that satisfies the constraint."""
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        
        # If a subset of variables is specified, make sure we only consider those
        unassigned = self.get_unassigned()
        if subset_vars is not None:
            subset_vars = set(subset_vars) & unassigned
            vars_to_assign = subset_vars
        else:
            vars_to_assign = unassigned
        
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            return assignment.copy() if self.evaluate(assignment) else None
        
        # Generate all possible solutions given the current assignment
        valid_solutions = list(self._validate_partial_assignment(assignment))
        
        if not valid_solutions:
            return None
        
        # Select a random solution
        solution = random.choice(valid_solutions)
        
        # Only keep assignments for the requested subset
        if subset_vars is not None:
            result = assignment.copy()
            for var in subset_vars:
                idx = self.variable_order.index(var)
                result[var] = solution[idx]
            return result
        
        # Convert the solution to an assignment
        result = assignment.copy()
        for i, val in enumerate(solution):
            var = self.variable_order[i]
            if var in vars_to_assign:
                result[var] = val
        
        return result
    
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint."""
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        
        # If a subset of variables is specified, make sure we only consider those
        unassigned = self.get_unassigned()
        if subset_vars is not None:
            subset_vars = set(subset_vars) & unassigned
            vars_to_assign = subset_vars
        else:
            vars_to_assign = unassigned
        
        # If there's nothing to assign, check if the constraint is satisfied
        if not vars_to_assign:
            if self.evaluate(assignment):
                yield assignment.copy()
            return
        
        # Generate all possible valid sequences
        for solution in self._validate_partial_assignment(assignment):
            # Convert the solution to an assignment
            result = assignment.copy()
            
            if subset_vars is not None:
                # Only include the subset variables in the result
                for var in vars_to_assign:
                    idx = self.variable_order.index(var)
                    result[var] = solution[idx]
            else:
                # Include all unassigned variables
                for i, val in enumerate(solution):
                    var = self.variable_order[i]
                    if var in unassigned:
                        result[var] = val
            
            yield result
    
    def _validate_partial_assignment(self, partial_assignment):
        """Generate all valid sequences that satisfy the run lengths and match the partial assignment."""
        # Create a binary sequence with None for unassigned positions
        sequence = []
        for v in self.variable_order:
            if v in partial_assignment:
                sequence.append(partial_assignment[v])
            elif v.value is not None:
                sequence.append(v.value)
            else:
                sequence.append(None)
        
        # Generate all valid solutions using a recursive approach
        yield from self._generate_valid_sequences(sequence, 0, self.run_lengths)
    
    def _generate_valid_sequences(self, sequence, pos, remaining_runs):
        """Recursively generate all valid sequences from a given position."""
        n = len(sequence)
        
        # Base case: if no more runs to place
        if not remaining_runs:
            # Check if the rest of the sequence can be filled with zeros
            # without creating any additional runs
            valid = True
            for i in range(pos, n):
                if sequence[i] == 1:
                    valid = False
                    break
            
            if valid:
                # Create a complete sequence with all remaining positions as 0
                result = sequence.copy()
                for i in range(n):
                    if result[i] is None:
                        result[i] = 0
                
                # Verify the final sequence has exactly the expected runs
                runs = []
                current_run = 0
                for bit in result:
                    if bit == 1:
                        current_run += 1
                    elif current_run > 0:
                        runs.append(current_run)
                        current_run = 0
                if current_run > 0:
                    runs.append(current_run)
                
                if runs == self.run_lengths:
                    yield result
            return
        
        # Get the next run length
        run_length = remaining_runs[0]
        new_remaining = remaining_runs[1:]
        
        # Calculate the maximum position where this run can start
        # Need to fit: current run + at least one space after it + all remaining runs + spaces between them
        remaining_space_needed = sum(new_remaining) + len(new_remaining)
        max_start_pos = n - run_length - remaining_space_needed
        
        # Try placing the current run at each possible position
        for start_pos in range(pos, max_start_pos + 1):
            # Check if we can place a zero before the run (if start_pos > pos)
            if start_pos > pos:
                # Must have a zero before the run or be at the beginning
                if sequence[start_pos - 1] == 1:
                    continue
            
            # Check if the run doesn't conflict with fixed positions
            valid = True
            for i in range(start_pos, start_pos + run_length):
                if i < n and sequence[i] == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if we can place a zero after the run (if required)
            end_pos = start_pos + run_length
            if end_pos < n:
                if sequence[end_pos] == 1:
                    continue
                end_pos += 1  # Skip the mandatory space after the run
            
            # Create a new sequence with this run placed
            new_sequence = sequence.copy()
            
            # Fill zeros before the run
            for i in range(pos, start_pos):
                if new_sequence[i] is None:
                    new_sequence[i] = 0
            
            # Fill the run with ones
            for i in range(start_pos, start_pos + run_length):
                if i < n:
                    new_sequence[i] = 1
            
            # Place a zero after the run if needed
            if start_pos + run_length < n:
                new_sequence[start_pos + run_length] = 0
            
            # Recursively fill the rest of the sequence
            yield from self._generate_valid_sequences(new_sequence, end_pos, new_remaining)




if __name__ == "__main__":
    from utils.assignment_utils import print_assignments
    v0 = Variable("v0", domain={0,1})
    v1 = Variable("v1", domain={0,1})
    v2 = Variable("v2", domain={0,1})
    v3 = Variable("v3", domain={0,1})
    v4 = Variable("v4", domain={0,1})
    v5 = Variable("v5", domain={0,1})

    initial_assignments = {v1:1}

    constraint = RunLengthsConstraint({v0,v1,v2,v3,v4,v5}, [2], 
                                      variable_order=[v0,v1,v2,v3,v4,v5])
    print(constraint)
    print("Partial assignment:", initial_assignments)
    print("Assignments:")


    print_assignments(list(constraint.possible_solutions(partial_assignment=initial_assignments)), 10)

