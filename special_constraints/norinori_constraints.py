import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints import Constraint, Variable
import random
from math import comb
from grammar import Sum, Number
from utils.assignment_utils import print_assignments


class DuoConstraint(Constraint):
    """
    Represents a constraint where if variable v is 1, then exactly one variable in V_S must be 1.
    This is an implication constraint: v => (exactly one of v_1, v_2, ..., v_n is 1) 
    where V_S = {v_1, v_2, ..., v_n}.
    """
    
    def __init__(self, v, v_s, **kwargs):
        """
        Initialize a DuoConstraint.
         
        Args:
            v: The antecedent variable (if this is 1...)
            v_s: Set of consequent variables (...then exactly one of these must be 1)
        """
        # Create the full set of variables including v and v_s
        variables = {v}.union(v_s)
        super().__init__(variables, 0, **kwargs)  # Target value not used in this constraint
        
        # Store the specific variables
        self.v = v
        self.v_s = set(v_s)
        
        # Error if v is in v_s
        if v in self.v_s:
            raise ValueError("Variable v cannot be in set V_S")
        
        # Check if all variables are binary
        for var in variables:
            if var.domain != {0, 1}:
                raise ValueError(f"All variables in DuoConstraint must be binary, got {var.domain}")

    def __str__(self) -> str:
        """String representation showing the implication relationship."""
        v_s_names = ", ".join(str(v) for v in sorted(self.v_s, key=lambda v: v.name))
        return f"If {self.v} then exactly one of [{v_s_names}] (p={self.get_p_correct():.2f})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(('duo', self.v, frozenset(self.v_s)))

    def __eq__(self, other):
        return (isinstance(other, DuoConstraint) and
                self.v == other.v and
                self.v_s == other.v_s)

    def copy(self):
        """Create a copy of this constraint."""
        return DuoConstraint(self.v, self.v_s.copy())
    
    def size(self):
        """Calculate the number of possible solutions to this constraint."""
        if self.test_contradiction():
            return 0
        
        unassigned = self.get_unassigned()
        
        # If no variables are unassigned, there's only one solution (current assignment)
        if not unassigned:
            return 1 if self.evaluate({var: var.value for var in self.variables}) else 0
        
        # If v is assigned to 0, any assignment to v_s works
        if self.v.value == 0:
            return 2 ** len(unassigned)
        
        # If v is assigned to 1, exactly one variable in v_s must be 1
        if self.v.value == 1:
            # Count how many variables in v_s are already assigned to 1
            assigned_v_s_ones = sum(1 for var in self.v_s if var.value == 1)
            
            if assigned_v_s_ones > 1:
                # Contradiction: more than one v_s is already 1
                return 0
                
            if assigned_v_s_ones == 1:
                # One v_s is already 1, all others must be 0
                # Check if any are already assigned to values that conflict
                if any(var.value == 1 for var in self.v_s if var.value is not None and var not in self.get_assigned()):
                    return 0
                    
                # All other unassigned v_s must be 0, and other variables can be anything
                unassigned_v_s = self.v_s & unassigned
                other_unassigned = unassigned - unassigned_v_s
                return 2 ** len(other_unassigned)
                
            # No v_s is assigned to 1 yet, exactly one must be
            unassigned_v_s = self.v_s & unassigned
            other_unassigned = unassigned - unassigned_v_s
            
            # Check if any v_s is already assigned to 0
            if any(var.value == 0 for var in self.v_s):
                # Some v_s are 0, exactly one of the unassigned must be 1
                return len(unassigned_v_s) * (2 ** len(other_unassigned))
            
            # No v_s is assigned yet, we have len(unassigned_v_s) choices for which one is 1
            return len(unassigned_v_s) * (2 ** len(other_unassigned))
        
        # v is unassigned
        # Calculate size for both v=0 and v=1 cases
        v_0_size = 2 ** (len(unassigned) - 1)  # v=0 case: all other variables can be anything
        
        # v=1 case: exactly one v_s must be 1
        unassigned_v_s = self.v_s & unassigned
        other_unassigned = unassigned - unassigned_v_s - {self.v}
        
        # Count how many v_s are already assigned to 1
        assigned_v_s_ones = sum(1 for var in self.v_s if var.value == 1)
        
        if assigned_v_s_ones > 1:
            # Contradiction: can't have v=1 in this case
            return v_0_size
        
        if assigned_v_s_ones == 1:
            # One v_s is already 1, all others must be 0
            # Check if any are already assigned and conflict
            if any(var.value == 1 for var in self.v_s if var not in self.get_assigned() and var.value is not None):
                return v_0_size
                
            # v=1 is valid, all unassigned v_s must be 0
            v_1_size = 2 ** len(other_unassigned)
        else:
            # No v_s is assigned to 1 yet, exactly one must be
            # Check if any v_s is already assigned to 0
            unassigned_count = len(unassigned_v_s)
            for var in self.v_s:
                if var.value == 0:
                    unassigned_count -= 1
                    
            v_1_size = unassigned_count * (2 ** len(other_unassigned))
            
        return v_0_size + v_1_size
    
    def initial_size(self):
        """Calculate the initial number of possible solutions before any assignments."""
        # Total possible assignments for n binary variables: 2^n
        n = len(self.variables)
        total = 2 ** n
        
        # Count valid assignments:
        # 1. v = 0: Any assignment to v_s (2^|v_s|)
        # 2. v = 1: Exactly one v_s is 1 (|v_s| possibilities)
        
        v_0_solutions = 2 ** len(self.v_s)  # When v=0, any assignment to v_s works
        v_1_solutions = len(self.v_s)       # When v=1, exactly one of v_s must be 1
        
        return v_0_solutions + v_1_solutions
    
    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy."""
        # If v is 0, constraint is automatically satisfied
        if self.v.value == 0:
            return False
            
        # If v is 1, exactly one variable in v_s must be 1
        if self.v.value == 1:
            # Count number of variables in v_s that are 1
            vs_ones_count = sum(1 for var in self.v_s if var.value == 1)
            
            # If more than one is 1, contradiction
            if vs_ones_count > 1:
                return True
                
            # If one is already 1, check if any others are not 0
            if vs_ones_count == 1:
                return any(var.value == 1 for var in self.v_s 
                          if var.value is not None and var not in self.get_assigned())
                
            # If all assigned v_s are 0, check if there are any unassigned ones
            vs_zeros_count = sum(1 for var in self.v_s if var.value == 0)
            if vs_zeros_count == len(self.v_s):
                # All v_s are 0, contradiction
                return True
                
        # No contradiction detected
        return False
    
    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution."""
        # Get the value of v from assignment or current value
        v_val = partial_assignment.get(self.v, self.v.value)
        
        # If v is 0 or unassigned, constraint is consistent
        if v_val != 1:
            return 1
            
        # If v is 1, exactly one variable in v_s must be 1
        # Count how many variables in v_s are assigned to 1
        vs_ones_count = 0
        for var in self.v_s:
            # Check partial assignment first, then current value
            if var in partial_assignment:
                vs_ones_count += partial_assignment[var]
            elif var.value == 1:
                vs_ones_count += 1
                
        # If already more than one is 1, inconsistent
        if vs_ones_count > 1:
            return 0
            
        # If one is already 1, all others must be 0
        if vs_ones_count == 1:
            for var in self.v_s:
                if var in partial_assignment:
                    if partial_assignment[var] != 0 and partial_assignment[var] != 1:
                        return 0
                elif var.value is not None and var.value != 0 and var.value != 1:
                    return 0
            return 1
            
        # If no variables are 1 yet, check if there are any that could be 1
        # Count how many variables in v_s are assigned to 0
        vs_zeros_count = 0
        for var in self.v_s:
            if var in partial_assignment and partial_assignment[var] == 0:
                vs_zeros_count += 1
            elif var.value == 0:
                vs_zeros_count += 1
                
        # If all variables are 0, inconsistent
        if vs_zeros_count == len(self.v_s):
            return 0
            
        # At least one variable could be 1
        return 1

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment."""
        # Get the value of v from assignment or current value
        v_val = assignment.get(self.v, self.v.value)
        
        # If v is 0, constraint is satisfied
        if v_val == 0:
            return 1
            
        # If v is 1, exactly one variable in v_s must be 1
        vs_ones_count = sum(assignment.get(var, var.value) == 1 for var in self.v_s)
        return 1 if vs_ones_count == 1 else 0

    def sample(self, partial_assignment=None, subset_vars=None):
        """Sample a random assignment that satisfies the constraint."""
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
        if not subset_vars:
            return assignment.copy() if self.evaluate(assignment) else None
        
        # First, determine the value of v (if it's in our subset)
        if self.v in subset_vars:
            # Decide if v should be 0 or 1
            v_val = random.choice([0, 1])
            assignment[self.v] = v_val
            subset_vars.remove(self.v)
        else:
            # v is already assigned or not in our subset
            v_val = assignment.get(self.v, self.v.value)
        
        # If v is 1, exactly one variable in v_s must be 1
        if v_val == 1:
            unassigned_v_s = self.v_s & subset_vars
            v_s_in_assignment = set(var for var in self.v_s if var in assignment)
            assigned_v_s = self.v_s - unassigned - v_s_in_assignment
            
            # Count how many v_s are already 1
            vs_ones_count = 0
            for var in assigned_v_s:
                if var.value == 1:
                    vs_ones_count += 1
                    
            for var in v_s_in_assignment:
                if assignment[var] == 1:
                    vs_ones_count += 1
            
            if vs_ones_count > 1:
                # Contradiction: more than one v_s is already 1
                return None
                
            if vs_ones_count == 1:
                # One is already 1, set all others to 0
                for var in unassigned_v_s:
                    assignment[var] = 0
            else:
                # No variable is 1 yet, choose one randomly and set it to 1
                # First check if any v_s is already assigned 0
                available_vars = []
                for var in unassigned_v_s:
                    available_vars.append(var)
                
                for var in assigned_v_s:
                    if var.value == 0:
                        available_vars = [v for v in available_vars if v != var]
                
                for var in v_s_in_assignment:
                    if assignment[var] == 0:
                        available_vars = [v for v in available_vars if v != var]
                
                if not available_vars:
                    # No variable can be set to 1
                    return None
                    
                # Choose one randomly to be 1
                chosen_var = random.choice(available_vars)
                assignment[chosen_var] = 1
                
                # Set all others to 0
                for var in unassigned_v_s:
                    if var != chosen_var:
                        assignment[var] = 0
        
        # For remaining variables not in v_s
        remaining_vars = subset_vars - self.v_s
        for var in remaining_vars:
            assignment[var] = random.choice([0, 1])
            
        return assignment

    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint."""
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
        if not subset_vars:
            yield assignment.copy()
            return
        
        # Convert subset_vars to a list for recursive enumeration
        vars_list = list(subset_vars)
        
        def generate_solutions(index, current_assignment):
            """Recursively generate all solutions that satisfy the constraint."""
            if index == len(vars_list):
                # We've assigned values to all variables in subset_vars
                if self.is_consistent(current_assignment):
                    yield current_assignment.copy()
                return
            
            var = vars_list[index]
            
            # Try setting var to 0
            current_assignment[var] = 0
            if self.is_consistent(current_assignment):
                yield from generate_solutions(index + 1, current_assignment)
            
            # Try setting var to 1
            current_assignment[var] = 1
            if self.is_consistent(current_assignment):
                yield from generate_solutions(index + 1, current_assignment)
            
            # Backtrack
            del current_assignment[var]
        
        yield from generate_solutions(0, assignment.copy())


if __name__ == "__main__":
    from constraints import PartialConstraint
    v0 = Variable("v0", domain={0,1})
    v1 = Variable("v1", domain={0,1})
    v2 = Variable("v2", domain={0,1})
    v3 = Variable("v3", domain={0,1})
    constraint = DuoConstraint(v0, {v1,v2})
    partial_assignment = {v0: 1}
    v1.assign(1)

    print(f"Constraint: {constraint}")
    print_assignments(list(constraint.possible_solutions()))

    print(f"Constraint: {constraint} with partial assignment: {partial_assignment}")
    print_assignments(list(constraint.possible_solutions(partial_assignment=partial_assignment)))
    
    subset_vars = {v0,v1}
    print(f"Constraint: {constraint} with partial assignment: {partial_assignment} and subset vars: {subset_vars}")
    print_assignments(list(constraint.possible_solutions(partial_assignment=partial_assignment, 
                                                         subset_vars=subset_vars)))
    
    subset_vars = {v0}
    print(f"Constraint: {constraint} with partial assignment: {partial_assignment} and subset vars: {subset_vars}")

    print_assignments(list(constraint.possible_solutions(partial_assignment=partial_assignment, 
                                                         subset_vars=subset_vars)))