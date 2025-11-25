import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random


from constraints import Constraint
from grammar import Variable
from special_constraints.generalized_constraints import GeneralEqualityConstraint

class ConnectivityConstraint(Constraint):
    """Constraint ensuring all islands are connected by bridges."""
    
    def __init__(self, islands, bridge_vars, **kwargs):
        """
        Initialize ConnectivityConstraint.
        
        Args:
            islands: List of islands, each as (r, c, value)
            bridge_vars: Dictionary mapping (island1, island2) -> Variable
        """
        self.islands = islands
        self.bridge_vars = bridge_vars
        
        # Extract just the coordinates for each island (remove the bridge count)
        self.island_coords = [(r, c) for r, c, _ in islands]
        
        variables = list(bridge_vars.values())
        # Pass target=0 to match the base Constraint class's signature
        super().__init__(variables, target=0, **kwargs)
        
        # Cache for component calculations
        self._component_cache = {}
        self._adjacency_cache = {}
        
        # Print a warning if there are no islands or bridge variables
        if not self.island_coords:
            print("WARNING: ConnectivityConstraint initialized with no islands")
        if not self.bridge_vars:
            print("WARNING: ConnectivityConstraint initialized with no bridge variables")

    def _get_adjacency_list(self, assignment):
        """Create an adjacency list for islands based on bridge assignments.
        
        Args:
            assignment: Dictionary of Variable -> value
            
        Returns:
            Dictionary mapping island coordinates to list of connected neighbors
        """
        # Create a key for caching based on assigned bridge variables
        cache_key = tuple(sorted((var.name, assignment.get(var, var.value)) 
                        for var in self.bridge_vars.values() 
                        if var in assignment or var.value is not None))
        
        # Return cached result if available
        if cache_key in self._adjacency_cache:
            return self._adjacency_cache[cache_key]
        
        # Create adjacency list
        adjacency = {island: [] for island in self.island_coords}
        
        # Add edges based on assigned bridges
        for (island1, island2), bridge_var in self.bridge_vars.items():
            value = assignment.get(bridge_var, bridge_var.value)
            if value is not None and value > 0:
                adjacency[island1].append(island2)
                adjacency[island2].append(island1)
        
        # Cache the result
        self._adjacency_cache[cache_key] = adjacency
        return adjacency

    def is_connected(self, assignment):
        """
        Check if islands form a single connected component.
        
        Args:
            assignment: Dictionary of Variable -> value
            
        Returns:
            True if all islands are connected, False otherwise
        """
        # Trivial cases
        if len(self.island_coords) <= 1:
            return True  # 0 or 1 islands are always connected
        
        # Get the adjacency list
        adjacency = self._get_adjacency_list(assignment)
        
        # Check for isolated islands (islands with no connections)
        isolated_islands = [island for island, neighbors in adjacency.items() if not neighbors]
        if isolated_islands:
            # At least one island is isolated
            return False
        
        # Use BFS to check if all islands are connected
        visited = set()
        queue = [self.island_coords[0]]  # Start from the first island
        
        while queue:
            current = queue.pop(0)
            visited.add(current)
            # Add all unvisited neighbors to the queue
            queue.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)
        
        # Check if all islands were visited
        return len(visited) == len(self.island_coords)

    def is_consistent(self, assignment, subset_vars=None):
        """
        Simpler version: check if there are fully assigned disconnected components.
        
        Args:
            assignment: Dictionary of Variable -> value
            subset_vars: Optional set of variables to consider
            
        Returns:
            1 if consistent, 0 if inconsistent
        """
        # Trivial cases
        if len(self.island_coords) <= 1:
            return 1  # 0 or 1 islands are always connected
        
        # Quick check: if we're working with a subset, be optimistic
        if subset_vars is not None:
            # Check if we're missing variables that could connect components
            remaining_vars = self.variables - set(subset_vars if subset_vars else [])
            if remaining_vars:
                # If there are remaining variables, be optimistic about connectivity
                return 1
        
        # Get the adjacency list for assigned bridges
        adjacency = self._get_adjacency_list(assignment)
        
        # Find connected components using BFS
        components = []
        visited = set()
        
        for island in self.island_coords:
            if island in visited:
                continue
            
            # Start a new component
            component = set()
            queue = [island]
            
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    # Add all unvisited neighbors to the queue
                    queue.extend(neighbor for neighbor in adjacency.get(current, []) 
                              if neighbor not in visited)
            
            components.append(component)
        
        # If all islands are already connected, the assignment is consistent
        if len(components) == 1:
            return 1
        
        # Check if all bridge variables are assigned
        # If not, we can be optimistic about connectivity
        all_assigned = True
        
        for bridge_var in self.bridge_vars.values():
            # Consider only relevant variables
            if subset_vars is not None and bridge_var not in subset_vars:
                continue
                
            # Check if this variable is unassigned
            if bridge_var not in assignment and bridge_var.value is None:
                all_assigned = False
                break
        
        # If not all bridges are assigned, be optimistic
        if not all_assigned:
            return 1
            
        # If we have multiple components and all bridges are assigned,
        # then the assignment is inconsistent
        return 0

    def possible_solutions(self, partial_assignment=None, subset_vars=None, max_solutions=None):
        """
        Generate all possible assignments that satisfy connectivity.
        
        Args:
            partial_assignment: Dictionary of Variable -> value to start with
            subset_vars: Optional set of variables to consider
            max_solutions: Maximum number of solutions to return
            
        Yields:
            Complete assignments that satisfy connectivity
        """
        # Clear caches before starting a new search
        self._component_cache = {}
        self._adjacency_cache = {}
        
        assignment = partial_assignment if partial_assignment is not None else {}
        
        # Check if the partial assignment is already inconsistent
        if not self.is_consistent(assignment, subset_vars):
            return  # No solutions possible
        
        # Create a copy to avoid modifying the input assignment
        current_assignment = assignment.copy()
        
        # Get unassigned variables
        unassigned = []
        for var in self.variables:
            if var not in current_assignment and var.value is None:
                if subset_vars is None or var in subset_vars:
                    unassigned.append(var)
        
        # Sort variables by their potential impact on connectivity
        unassigned = self._sort_by_connectivity_impact(unassigned, current_assignment)
        
        # Counter for limiting solutions
        solutions_count = 0
        
        def backtrack(idx):
            nonlocal solutions_count
            
            # Base case: all variables assigned
            if idx >= len(unassigned):
                # Only check if current assignment has already made islands disconnected
                # If it's still possibly connectable, we assume the remaining vars could connect them
                if self.is_consistent(current_assignment, subset_vars) == 1:
                    yield current_assignment.copy()
                    solutions_count += 1
                return
            
            # Check if we've reached the solution limit
            if max_solutions is not None and solutions_count >= max_solutions:
                return
            
            # Get the current variable
            var = unassigned[idx]
            
            # Try all possible values in the domain
            # Try higher values first since they're more likely to create connectivity
            domain_values = sorted(var.domain, reverse=True)
            for value in domain_values:
                current_assignment[var] = value
                
                # Check if the partial assignment is consistent before recursing
                if self.is_consistent(current_assignment, subset_vars) == 1:
                    yield from backtrack(idx + 1)
                
                # If we've reached the solution limit, stop trying values
                if max_solutions is not None and solutions_count >= max_solutions:
                    break
            
            # Backtrack by removing the variable from the assignment
            del current_assignment[var]
        
        # Start the backtracking process
        yield from backtrack(0)
    
    def _sort_by_connectivity_impact(self, variables, assignment):
        """Sort variables by their potential impact on connectivity."""
        # Get components for current assignment
        adjacency = self._get_adjacency_list(assignment)
        
        # Find current components
        components = {}
        next_component = 0
        visited = set()
        
        # Use BFS to find components
        for island in self.island_coords:
            if island in visited:
                continue
                
            # Start a new component
            component = set([island])
            queue = [island]
            visited.add(island)
            
            while queue:
                current = queue.pop(0)
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            
            # Assign component ID to all islands in this component
            for island in component:
                components[island] = next_component
            
            next_component += 1
        
        # Score variables by their potential to connect different components
        var_scores = {}
        for var in variables:
            # Skip if already assigned
            if var in assignment or var.value is not None:
                var_scores[var] = -1  # Lowest priority
                continue
                
            score = 0
            # Find the islands this bridge connects
            for (island1, island2), bridge_var in self.bridge_vars.items():
                if bridge_var == var:
                    # Find which components these islands belong to
                    comp1 = components.get(island1, -1)
                    comp2 = components.get(island2, -1)
                    
                    if comp1 != comp2:
                        # Connecting different components is highest priority
                        score = 100
                    elif comp1 == -1 or comp2 == -1:
                        # Connecting to an isolated island is next priority
                        score = 50
                    else:
                        # Creating redundant connections is lowest priority
                        score = 10
                        
                    # Add bridge count to the score (prefer higher values)
                    max_value = max(var.domain) if var.domain else 0
                    score += max_value
                    
                    break
            
            var_scores[var] = score
        
        # Return variables sorted by score (highest to lowest)
        return sorted(variables, key=lambda var: var_scores.get(var, 0), reverse=True)
    
    def evaluate(self, assignment):
        """Evaluate if the assignment satisfies connectivity."""
        return 1 if self.is_connected(assignment) else 0
    
    def __str__(self):
        """String representation of the constraint."""
        return f"ConnectivityConstraint(islands={len(self.island_coords)}, bridges={len(self.bridge_vars)})"

    def test_contradiction(self):
        """Test if current assignments make connectivity impossible."""
        # Create a partial assignment from assigned variables
        partial_assignment = {v: v.value for v in self.get_assigned() if v.value is not None}
        
        # Use is_consistent to check for contradictions
        return self.is_consistent(partial_assignment) == 0
    
    def size(self):
        """Estimate the number of valid connected solutions."""
        # If we have a contradiction, size is 0
        if self.test_contradiction():
            return 0
            
        # Count the number of unassigned variables
        unassigned = self.get_unassigned()
        n_unassigned = len(unassigned)
        
        if n_unassigned == 0:
            # If all variables are assigned, evaluate current assignment
            assignment = {v: v.value for v in self.variables if v.value is not None}
            return 1 if self.is_connected(assignment) else 0
            
        # For small numbers of unassigned variables, use a cache-friendly approach
        if n_unassigned <= 4:
            # Use our cached is_consistent to avoid recalculating components
            # Calculate an approximation by sampling the solution space
            domain_sizes = {var: len(var.domain) for var in unassigned}
            total_combinations = 1
            for size in domain_sizes.values():
                total_combinations *= size
                
            # For very small spaces, enumerate all solutions
            if total_combinations <= 100:
                solutions = list(self.possible_solutions(max_solutions=total_combinations))
                return len(solutions)
                
            # For larger spaces, sample and estimate
            # The approach is to sample ~sqrt(total_combinations) solutions
            # and estimate the proportion that are connected
            sample_size = min(int(total_combinations**0.5) + 5, 50)
            
            # Generate sample_size random combinations
            valid_count = 0
            for _ in range(sample_size):
                test_assignment = {v: v.value for v in self.get_assigned() if v.value is not None}
                for var in unassigned:
                    test_assignment[var] = random.choice(list(var.domain))
                
                if self.is_connected(test_assignment):
                    valid_count += 1
            
            # Calculate the estimated proportion of valid solutions
            if sample_size > 0:
                proportion = valid_count / sample_size
                return max(1, int(proportion * total_combinations))
            else:
                return 1  # Default to 1 if we couldn't sample
        
        # For larger problems, approximate
        # Analyze the component structure to make a better estimate
        partial_assignment = {v: v.value for v in self.get_assigned() if v.value is not None}
        adjacency = self._get_adjacency_list(partial_assignment)
        
        # Count components
        components = []
        visited = set()
        
        for island in self.island_coords:
            if island in visited:
                continue
            
            # Start a new component
            component = set()
            queue = [island]
            
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    
                    # Add all unvisited neighbors to the queue
                    queue.extend(neighbor for neighbor in adjacency.get(current, []) if neighbor not in visited)
            
            components.append(component)
        
        num_components = len(components)
        
        # Calculate how many potential bridges exist between components
        bridge_count = 0
        for var in unassigned:
            # Find the bridge endpoints
            for (i1, i2), bridge_var in self.bridge_vars.items():
                if bridge_var == var:
                    # Find which components these islands belong to
                    comp1 = -1
                    comp2 = -1
                    for idx, comp in enumerate(components):
                        if i1 in comp:
                            comp1 = idx
                        if i2 in comp:
                            comp2 = idx
                    
                    # If this bridge connects different components, count it
                    if comp1 != comp2 and comp1 >= 0 and comp2 >= 0:
                        bridge_count += 1
                    break
        
        # Base case: If there's only one component already
        if num_components <= 1:
            # For a fully connected graph, more solutions are likely valid
            total_combinations = 1
            for var in unassigned:
                total_combinations *= len(var.domain)
            return max(1, int(total_combinations * 0.7))  # 70% estimated to be valid
            
        # If we need to connect components:
        # Penalty factor: estimate the likelihood of having enough bridges 
        # to connect all components based on the number of potential connecting bridges
        min_bridges_needed = num_components - 1
        
        if bridge_count < min_bridges_needed:
            # Not enough bridges to connect components
            return 0
            
        # The more excess bridges, the more likely we can connect all components
        # Calculation: 3^(total unassigned vars) * adjustment factor
        total_combinations = 3**n_unassigned  # Assuming domain size of 3 as typical
        
        # Calculate adjustment factor based on bridge count and components
        # More components = harder to connect = lower probability
        # More connecting bridges = easier to connect = higher probability
        adjustment_factor = (bridge_count / (min_bridges_needed * 2)) / (num_components ** 0.5)
        adjustment_factor = min(max(adjustment_factor, 0.05), 0.8)  # Clamp between 5% and 80%
        
        return max(1, int(total_combinations * adjustment_factor))
        
    def copy(self):
        """Create a copy of this constraint."""
        return ConnectivityConstraint(
            islands=self.islands.copy(),
            bridge_vars=self.bridge_vars.copy()
        )


class NandConstraint(Constraint):
    """Ensures that at least one of the given variables has a value of 0.
    
    This is a NAND constraint, which is satisfied when not all variables have non-zero values.
    It is useful for bridge puzzles to prevent certain combinations of bridges.
    """
    
    def __init__(self, variables, **kwargs):
        """Initialize a NAND constraint.
        
        Args:
            variables: Set of variables where at least one should have a value of 0
        """
        super().__init__(variables, target=1, **kwargs)
        
        # Verify all variables have compatible domains
        for var in variables:
            if 0 not in var.domain:
                raise ValueError(f"All variables in NandConstraint must have 0 in their domain, got {var.domain}")
    
    def __str__(self):
        """Return a string representation of the constraint."""
        var_names = [str(v) for v in sorted(self.variables, key=str)]
        return f"NAND({', '.join(var_names)})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(('nand', frozenset(self.variables)))
    
    def __eq__(self, other):
        return (isinstance(other, NandConstraint) and
                self.variables == other.variables)
    
    def evaluate(self, assignment):
        """Check if at least one variable has a value of 0 (NAND logic)."""
        all_non_zero = True
        for var in self.variables:
            value = assignment.get(var, var.value)
            if value is None or value == 0:
                all_non_zero = False
                break
        
        return 1 if not all_non_zero else 0
    
    def is_consistent(self, partial_assignment):
        """Check if the partial assignment could lead to a valid solution."""
        # Check if any variable is already assigned 0, which would satisfy NAND
        for var in self.variables:
            value = partial_assignment.get(var, var.value)
            if value is not None and value == 0:
                return 1
        
        # If all variables in the assignment are non-zero, we need at least one unassigned
        unassigned = self.get_unassigned() - set(partial_assignment.keys())
        if not unassigned:
            return 0  # All variables assigned non-zero, inconsistent
        
        # Check if any unassigned variable could be assigned 0
        for var in unassigned:
            if 0 in var.domain:
                return 1  # At least one variable could be 0
        
        # No variable can be 0, inconsistent
        return 0
    
    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy."""
        # Count non-zero assigned variables
        all_assigned_non_zero = True
        unassigned_exist = False
        
        for var in self.variables:
            if var.value is None:
                unassigned_exist = True
                continue
            if var.value == 0:
                all_assigned_non_zero = False
                break
        
        # If all assigned variables are non-zero and there are no unassigned variables,
        # then we have a contradiction
        return all_assigned_non_zero and not unassigned_exist
    
    def size(self):
        """Calculate the number of possible solutions to this constraint."""
        if self.test_contradiction():
            return 0
            
        # Count unassigned variables
        unassigned = self.get_unassigned()
        n_unassigned = len(unassigned)
        
        # Check if any already assigned variable is 0, which would satisfy NAND
        for var in self.get_assigned():
            if var.value == 0:
                # NAND is satisfied, so all possible combinations of unassigned variables are valid
                return self._count_all_combinations(unassigned)
        
        # If all assigned variables are non-zero, we need at least one unassigned to be 0
        
        # For a more accurate calculation, we'll manually count valid solutions
        # This could be optimized further with mathematical formulas if needed
        valid_count = 0
        assigned_values = {var: var.value for var in self.get_assigned()}
        
        # If we don't have many unassigned variables, count manually
        if n_unassigned <= 8:
            # Count assignments where at least one variable is 0
            for assignment in self.possible_solutions():
                valid_count += 1
            return valid_count
        
        # For larger problems, use the formula: total - (all non-zero)
        total_combinations = self._count_all_combinations(unassigned)
        all_non_zero_combinations = self._count_all_non_zero_combinations(unassigned)
        
        # Solutions = all possible combinations - all non-zero combinations
        return total_combinations - all_non_zero_combinations
    
    def _count_all_combinations(self, variables):
        """Count all possible combinations for a set of variables."""
        if not variables:
            return 1
        
        total = 1
        for var in variables:
            total *= len(var.domain)
        return total
    
    def _count_all_non_zero_combinations(self, variables):
        """Count combinations where all variables have non-zero values."""
        if not variables:
            return 1
        
        total = 1
        for var in variables:
            # Count values in the domain that are not 0
            non_zero_values = sum(1 for v in var.domain if v != 0)
            total *= non_zero_values
        return total
    
    def sample(self, partial_assignment=None, subset_vars=None):
        """Sample a random assignment that satisfies the constraint."""
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned() - set(assignment.keys())
        
        # If subset_vars is specified, only consider those variables
        if subset_vars is not None:
            subset_vars = set(subset_vars) & unassigned
            vars_to_assign = subset_vars
        else:
            vars_to_assign = unassigned
        
        # Check if constraint is already satisfied by having a 0 value
        for var in self.variables:
            value = assignment.get(var, var.value)
            if value is not None and value == 0:
                # NAND already satisfied, assign random values to remaining variables
                for var in vars_to_assign:
                    assignment[var] = random.choice(list(var.domain))
                return assignment
        
        # None of the current variables are 0, we need to set at least one to 0
        # First check if we have any unassigned variables
        if not vars_to_assign:
            return None  # Cannot satisfy constraint
            
        # Choose a random unassigned variable to set to 0
        chosen_var = random.choice(list(vars_to_assign))
        if 0 in chosen_var.domain:
            assignment[chosen_var] = 0
            
            # Assign random values to other variables
            remaining_vars = vars_to_assign - {chosen_var}
            for var in remaining_vars:
                assignment[var] = random.choice(list(var.domain))
                
            return assignment
        
        # If we can't set chosen_var to 0, try others
        for var in vars_to_assign:
            if 0 in var.domain:
                assignment[var] = 0
                
                # Assign random values to other variables
                remaining_vars = vars_to_assign - {var}
                for other_var in remaining_vars:
                    assignment[other_var] = random.choice(list(other_var.domain))
                    
                return assignment
        
        # Can't satisfy constraint
        return None
    
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint."""
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned() - set(assignment.keys())
        
        # If subset_vars is specified, only consider those variables
        if subset_vars is not None:
            subset_vars = set(subset_vars) & unassigned
            vars_to_assign = subset_vars
        else:
            vars_to_assign = unassigned
        
        # Helper function to recursively generate assignments
        def generate_assignments(remaining_vars, current_assignment):
            if not remaining_vars:
                # Check if the assignment satisfies NAND (at least one var is 0)
                if self.evaluate(current_assignment) == 1:
                    yield current_assignment.copy()
                return
            
            # Take next variable
            var = next(iter(remaining_vars))
            new_remaining = remaining_vars - {var}
            
            # Try each possible value for the variable
            for value in var.domain:
                current_assignment[var] = value
                # Continue generating if consistent
                if self.is_consistent(current_assignment) == 1:
                    yield from generate_assignments(new_remaining, current_assignment)
            
            # Backtrack
            del current_assignment[var]
        
        # Generate all valid assignments
        yield from generate_assignments(vars_to_assign, assignment.copy())
    
    def copy(self):
        """Create a copy of this constraint."""
        return NandConstraint(self.variables.copy())



if __name__ == "__main__":

    islands = [(0, 0, 2), (0, 2, 2), (2, 0, 2), (2, 2, 2)]  # Four islands in corners with value 2
    
    # Create bridge variables with domain {0,1,2} (no bridge, single bridge, double bridge)
    v1 = Variable("v1", domain={0, 1, 2})  # bridge from (0,0) to (0,2)
    v2 = Variable("v2", domain={0, 1, 2})  # bridge from (0,0) to (2,0)
    v3 = Variable("v3", domain={0, 1, 2})  # bridge from (0,2) to (2,2)
    v4 = Variable("v4", domain={0, 1, 2})  # bridge from (2,0) to (2,2)
    
    # Map each bridge to its endpoints
    bridge_vars = {
        ((0, 0), (0, 2)): v1,
        ((0, 0), (2, 0)): v2,
        ((0, 2), (2, 2)): v3,
        ((2, 0), (2, 2)): v4
    }
    
    # Create connectivity constraint
    connectivity = ConnectivityConstraint(islands, bridge_vars)
    
    print("Testing ConnectivityConstraint implementation:")
    print(f"Islands: {islands}")
    print(f"Bridge variables: {bridge_vars}")
    
    # Print a visual representation of the islands and bridges
    print("\nIsland layout (grid coordinates):")
    print("(0,0) --- v1 --- (0,2)")
    print("  |                |")
    print("  v2               v3")
    print("  |                |")
    print("(2,0) --- v4 --- (2,2)")
    print()
    
    # Test 1: All bridges with value 0 should not be connected
    test_assignment = {v1: 0, v2: 0, v3: 0, v4: 0}
    print(f"Test 1 - All bridges 0: {connectivity.evaluate(test_assignment)} (should be 0)")
    
    # Test 2: Creating a connected path should work
    test_assignment = {v1: 1, v2: 1, v3: 1, v4: 0}
    print(f"Test 2 - Connected path: {connectivity.evaluate(test_assignment)} (should be 1)")
    
    # Test 3: Disconnected subgraphs should fail
    test_assignment = {v1: 1, v2: 0, v3: 0, v4: 1}
    print(f"Test 3 - Disconnected subgraphs: {connectivity.evaluate(test_assignment)} (should be 0)")
    
    # Test 4: Different bridge values should still work if connected
    test_assignment = {v1: 2, v2: 1, v3: 1, v4: 0}
    print(f"Test 4 - Different bridge values in connected config: {connectivity.evaluate(test_assignment)} (should be 1)")
    
    # Test 5: Tree-shaped connected configuration without cycles
    test_assignment = {v1: 1, v2: 1, v3: 1, v4: 0}
    print(f"Test 5 - Tree configuration: {connectivity.evaluate(test_assignment)} (should be 1)")
    
    print("\nGenerating all valid connected configurations:")
    count = 0
    for solution in connectivity.possible_solutions():
        count += 1
        if count <= 10:  # Only print the first 10 solutions
            print(f"Solution {count}: " + 
                  f"v1={solution[v1]}, v2={solution[v2]}, v3={solution[v3]}, v4={solution[v4]}")
    
    print(f"\nTotal valid connected configurations: {count}")
    print(f"Estimated size via size() method: {connectivity.size()}")
    
    # Test partial assignments
    print("\nTesting with partial assignment {v1: 1}:")
    count = 0
    for solution in connectivity.possible_solutions(partial_assignment={v1: 1}):
        count += 1
        if count <= 5:
            print(f"Solution {count}: " + 
                  f"v1={solution[v1]}, v2={solution[v2]}, v3={solution[v3]}, v4={solution[v4]}")
    
    print(f"Total solutions with v1=1: {count}")
    
    # Test is_consistent method
    print("\nTesting is_consistent method:")
    print(f"Empty assignment: {connectivity.is_consistent({})} (should be 1)")
    print(f"v1=1, v2=0, v3=0, v4=0: {connectivity.is_consistent({v1: 1, v2: 0, v3: 0, v4: 0})} (should be 0)")
    print(f"v1=1, v2=1: {connectivity.is_consistent({v1: 1, v2: 1})} (should be 1)")

    # Now test the NandConstraint
    print("\n\nTesting NandConstraint implementation:")
    
    # Create variables with domain {0,1,2}
    n1 = Variable("n1", domain={0, 1, 2})
    n2 = Variable("n2", domain={0, 1, 2})
    n3 = Variable("n3", domain={0, 1, 2})
    
    # Create NAND constraint
    nand = NandConstraint({n1, n2, n3})
    
    print(f"NAND constraint: {nand}")
    
    # Test cases for NAND
    test_cases = [
        {n1: 0, n2: 0, n3: 0},  # Should be 1 (at least one is 0)
        {n1: 0, n2: 1, n3: 2},  # Should be 1 (at least one is 0)
        {n1: 1, n2: 0, n3: 2},  # Should be 1 (at least one is 0)
        {n1: 1, n2: 2, n3: 0},  # Should be 1 (at least one is 0)
        {n1: 1, n2: 1, n3: 1},  # Should be 0 (all are non-zero)
        {n1: 2, n2: 1, n3: 2},  # Should be 0 (all are non-zero)
    ]
    
    for i, test in enumerate(test_cases):
        result = nand.evaluate(test)
        expected = 1 if any(v == 0 for v in test.values()) else 0
        print(f"Test {i+1} - {test}: {result} (should be {expected})")
    
    # Test partial assignment consistency
    partial_tests = [
        {},  # Empty assignment should be consistent
        {n1: 0},  # One zero value should be consistent
        {n1: 1},  # One non-zero value should be consistent
        {n1: 1, n2: 1},  # Two non-zero values should be consistent
        {n1: 1, n2: 1, n3: 1},  # All non-zero values should be inconsistent
    ]
    
    print("\nTesting is_consistent with partial assignments:")
    for i, test in enumerate(partial_tests):
        result = nand.is_consistent(test)
        print(f"Partial test {i+1} - {test}: {result} (should be {1 if len(test) < 3 or any(v == 0 for v in test.values()) else 0})")
    
    # Test generating all solutions
    print("\nGenerating all valid NAND configurations:")
    count = 0
    for solution in nand.possible_solutions():
        count += 1
        if count <= 10:  # Only print the first 10 solutions
            print(f"Solution {count}: " + 
                  f"n1={solution[n1]}, n2={solution[n2]}, n3={solution[n3]}")
    
    print(f"\nTotal valid NAND configurations: {count}")
    print(f"Estimated size via size() method: {nand.size()}")