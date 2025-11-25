import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints import Constraint, Variable, InequalityConstraint
import random
from math import comb
from grammar import Sum, Number
from utils.assignment_utils import *
import numpy as np

def generate_polyominoes(n):
    """
    Generate all unique free polyominoes of size n.
    
    Args:
        n: The size of polyominoes to generate
        
    Returns:
        A list of polyominoes, each represented as a frozenset of (x, y) coordinates
    """
    # Initialize cache if not already done
    if not hasattr(generate_polyominoes, "cache"):
        generate_polyominoes.cache = {1: [frozenset([(0, 0)])]}
    
    # Return cached result if available
    if n in generate_polyominoes.cache:
        return generate_polyominoes.cache[n]
    
    # Base cases
    if n <= 0:
        return []
    
    # Get polyominoes of size n-1
    smaller_polys = generate_polyominoes(n-1)
    
    # Set to store unique polyominoes
    unique_polys = set()
    
    # Generate new polyominoes by adding one square
    for poly in smaller_polys:
        # Find adjacent positions
        adjacents = set()
        for x, y in poly:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (x + dx, y + dy)
                if new_pos not in poly:
                    adjacents.add(new_pos)
        
        # Add each adjacent position and canonicalize
        for adj in adjacents:
            new_poly = poly.union({adj})
            canonical = get_canonical_form(new_poly)
            unique_polys.add(canonical)
    
    # Cache and return results
    result = list(unique_polys)
    generate_polyominoes.cache[n] = result
    return result

def get_canonical_form(poly):
    """
    Get the canonical representation of a polyomino.
    
    Args:
        poly: A frozenset of (x, y) coordinates
        
    Returns:
        The canonical form as a frozenset
    """
    min_representation = None
    
    # Define the 8 transformations (4 rotations × 2 reflections)
    transformations = [
        lambda x, y: (x, y),               # Identity
        lambda x, y: (-y, x),              # 90° clockwise
        lambda x, y: (-x, -y),             # 180°
        lambda x, y: (y, -x),              # 270° clockwise
        lambda x, y: (-x, y),              # Reflect across y-axis
        lambda x, y: (y, x),               # Reflect across y=x
        lambda x, y: (x, -y),              # Reflect across x-axis
        lambda x, y: (-y, -x)              # Reflect across y=-x
    ]
    
    # Find the minimum representation
    for transform in transformations:
        # Apply transformation
        transformed = [transform(x, y) for x, y in poly]
        
        # Normalize to origin
        min_x = min(x for x, y in transformed)
        min_y = min(y for x, y in transformed)
        normalized = frozenset((x - min_x, y - min_y) for x, y in transformed)
        
        # Update the minimum representation
        if min_representation is None or normalized < min_representation:
            min_representation = normalized
    
    return min_representation


def visualize_3x3_grid(grid_values):
    """
    Visualize a 3x3 grid with water (0) and island (1) cells.
    
    Args:
        grid_values: A 3x3 list of 0s and 1s where:
            0 = water/black cells
            1 = island/white cells
    """
    result = []
    
    # Add separator
    separator = " +-----+"
    result.append(separator)
    
    # Process each row
    for row in grid_values:
        row_values = [("□ " if val == 0 else "■ ") for val in row] # □=water(0), ■=island(1)
        row_str = "| " + "".join(row_values) + "|"
        result.append(row_str)
    
    # Add bottom separator
    result.append(separator)
    
    return "\n".join(result)


def is_valid_water_pattern(grid_values):
    """
    Check if a 3x3 grid has a valid water (0) connectivity pattern.
    
    A valid pattern has:
    1. All water cells are orthogonally connected
    2. No 2x2 pools of water
    
    Args:
        grid_values: A 3x3 list of 0s and 1s
        
    Returns:
        True if the water pattern is valid, False otherwise
    """
    # Find all water cells (0)
    water_cells = []
    for r in range(3):
        for c in range(3):
            if grid_values[r][c] == 0:
                water_cells.append((r, c))
    
    if not water_cells:
        return True
    
    # If only one water cell, it's isolated and invalid
    if len(water_cells) == 1:
        return False
    
    # Check for 2x2 water pools
    for r in range(2):
        for c in range(2):
            if (grid_values[r][c] == 0 and grid_values[r][c+1] == 0 and
                grid_values[r+1][c] == 0 and grid_values[r+1][c+1] == 0):
                return False
    
    # Check connectivity of water cells using BFS
    visited = set()
    queue = [water_cells[0]]
    
    while queue:
        r, c = queue.pop(0)
        if (r, c) in visited:
            continue
            
        visited.add((r, c))
        
        # Check orthogonal neighbors
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3 and grid_values[nr][nc] == 0 and (nr, nc) not in visited:
                queue.append((nr, nc))
    
    # Check if all water cells were visited
    return len(visited) == len(water_cells)


def generate_valid_water_patterns():
    """
    Generate all valid 3x3 water patterns for Nurikabe.
    
    Returns:
        List of valid 3x3 grids where water (0) forms valid connected paths
    """
    valid_patterns = []
    
    # Generate all possible 3x3 grids (2^9 = 512 possibilities)
    for i in range(512):
        grid = []
        for r in range(3):
            row = []
            for c in range(3):
                # Extract bit from i
                bit_position = r * 3 + c
                bit = (i >> bit_position) & 1
                row.append(bit)
            grid.append(row)
        
        # Check if the pattern is valid
        if is_valid_water_pattern(grid):
            valid_patterns.append(grid)
    
    return valid_patterns


class PolyominoConstraint(Constraint):
    """Constraint ensuring that a numbered cell forms part of a polyomino (island) 
    of the specified size in Nurikabe puzzles.
    
    The polyomino must:
    - Be connected
    - Include the numbered cell
    - Have size equal to the number on the cell
    - Consist only of white cells ("land")
    """
    
    def __init__(self, variables, numbered_cell_pos, number_value, grid_size, all_numbered_cells=None, **kwargs):
        """Initialize a polyomino constraint for Nurikabe.
        
        Args:
            variables: Set of variables in the potential reach of the numbered cell
            numbered_cell_pos: (row, col) of the numbered cell
            number_value: Value of the numbered cell (size of required polyomino)
            grid_size: (rows, cols) of the grid
            all_numbered_cells: List of all numbered cell positions in the puzzle
        """
        # Filter variables to only include those within reach + 1 (for water buffer)
        numbered_row, numbered_col = numbered_cell_pos
        max_distance = number_value  # The numbered cell value determines the reach
        
        # Keep only variables within max_distance + 1 (for water buffer)
        filtered_variables = set()
        for var in variables:
            # Calculate Manhattan distance from numbered cell
            distance = abs(var.row - numbered_row) + abs(var.col - numbered_col)
            # Keep if within reach or one step beyond (for water buffer)
            if distance <= max_distance + 1:
                filtered_variables.add(var)
        
        super().__init__(filtered_variables, number_value, **kwargs)
        self.numbered_cell_pos = numbered_cell_pos
        self.number_value = number_value
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        
        # Store all numbered cells for checking adjacency rules
        self.all_numbered_cells = set() if all_numbered_cells is None else set(all_numbered_cells)
        # Remove this constraint's numbered cell from the set
        self.other_numbered_cells = self.all_numbered_cells - {self.numbered_cell_pos}
        
        # Build a mapping of positions to variables
        self.pos_to_var = {}
        for var in filtered_variables:
            self.pos_to_var[(var.row, var.col)] = var
            
        # Identify positions this constraint cares about
        self.positions = set(self.pos_to_var.keys()) | {self.numbered_cell_pos}
        
        # Cache valid polyominoes
        self._valid_polyominoes = None
        
    def __str__(self):
        return f"PolyominoConstraint: Cell {self.numbered_cell_pos} forms island of size {self.number_value}"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(('polyomino', self.numbered_cell_pos, self.number_value, frozenset(self.variables)))
    
    def __eq__(self, other):
        return (isinstance(other, PolyominoConstraint) and 
                self.numbered_cell_pos == other.numbered_cell_pos and
                self.number_value == other.number_value and
                self.variables == other.variables)
    
    def get_valid_polyominoes(self):
        """Generate all valid polyominoes of the required size containing the numbered cell."""
        if self._valid_polyominoes is not None:
            return self._valid_polyominoes
        
        # Start with just the numbered cell
        polyominoes = []
        
        # Special case for size 1
        if self.number_value == 1:
            polyominoes.append({self.numbered_cell_pos})
            self._valid_polyominoes = polyominoes
            return polyominoes
        
        # For larger sizes, do a BFS to find all possible polyominoes
        start = {self.numbered_cell_pos}
        queue = [start]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if frozenset(current) in visited:
                continue
                
            visited.add(frozenset(current))
            
            # If we have the right size, add to results
            if len(current) == self.number_value:
                polyominoes.append(current)
                continue
                
            # If too large, skip
            if len(current) > self.number_value:
                continue
            
            # Try adding each adjacent cell
            for pos in current:
                row, col = pos
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
                    nr, nc = row + dr, col + dc
                    new_pos = (nr, nc)
                    
                    # Skip if out of bounds or already in polyomino
                    if (not (0 <= nr < self.rows and 0 <= nc < self.cols) or 
                        new_pos in current or 
                        new_pos not in self.positions):
                        continue
                    
                    # Add the new position
                    new_poly = current.copy()
                    new_poly.add(new_pos)
                    queue.append(new_poly)
        
        self._valid_polyominoes = polyominoes
        return polyominoes
    
    def copy(self):
        """Create a copy of this constraint."""
        return PolyominoConstraint(
            self.variables.copy(), 
            self.numbered_cell_pos, 
            self.number_value, 
            self.grid_size,
            all_numbered_cells=self.all_numbered_cells
        )
    
    def size(self):
        """Calculate the number of possible solutions."""
        # Fast rejection if contradiction exists
        if self.test_contradiction():
            return 0
        
        # Get assigned variables
        assigned = self.get_assigned()
        
        # Fast path if all variables are assigned
        if not self.get_unassigned():
            assignment = {var: var.value for var in assigned}
            return 1 if self.evaluate(assignment) else 0
        
        # Use caching for repeated calculations
        if not hasattr(self, '_size_cache'):
            self._size_cache = {}
        
        # Create a cache key based on assigned variables
        cache_key = frozenset((var, var.value) for var in assigned)
        
        # Return cached result if available
        if cache_key in self._size_cache:
            return self._size_cache[cache_key]
        
        # Count solutions
        count = sum(1 for _ in self.possible_solutions())
        
        # Cache the result
        self._size_cache[cache_key] = count
        
        return count
    
    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy."""
        # Get currently assigned variables
        assigned = self.get_assigned()
        if not assigned:
            return False  # No assignments, so no contradiction
        
        # Create a partial assignment from currently assigned variables
        partial_assignment = {var: var.value for var in assigned}
        
        # Use is_consistent to determine if there's a contradiction
        return self.is_consistent(partial_assignment) == 0
    
    def find_island(self, assignment, start_pos):
        """Find all cells that are part of the island containing the start position.
        
        An island consists of orthogonally connected land cells (value=1).
        
        Args:
            assignment: Dictionary mapping variables to values
            start_pos: Starting position (row, col)
            
        Returns:
            Set of positions that form the island
        """
        island = set()
        queue = [start_pos]
        
        while queue:
            pos = queue.pop(0)
            if pos in island:
                continue
            
            # Add to island (for numbered cell or land cell)
            island.add(pos)
            
            # Check orthogonal neighbors
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                new_pos = (nr, nc)
                
                # Skip if out of range or already visited
                if (not (0 <= nr < self.rows and 0 <= nc < self.cols) or 
                    new_pos in island):
                    continue
                
                # Check if it's land (value=1) or the numbered cell itself
                if new_pos == self.numbered_cell_pos:
                    queue.append(new_pos)
                elif new_pos in self.positions:
                    var = self.pos_to_var.get(new_pos)
                    if var and assignment.get(var, None) == 1:  # Land
                        queue.append(new_pos)
        
        return island

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment.
        
        The constraint is satisfied if:
        1. The numbered cell forms part of an island of exactly the required size
        2. The island cells are all land (value=1)
        3. All cells adjacent to the island are water (value=0)
        4. The island is not adjacent to any other numbered cells
        """
        # Find the island containing the numbered cell
        island = self.find_island(assignment, self.numbered_cell_pos)
        
        # Check if island has the correct size
        if len(island) != self.number_value:
            return 0
        
        # Check all cells adjacent to the island are water (value=0)
        # and no other numbered cells are adjacent to this island
        adjacent = set()
        for pos in island:
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                new_pos = (nr, nc)
                
                if (0 <= nr < self.rows and 0 <= nc < self.cols and
                    new_pos not in island):
                    
                    # Check if adjacent to another numbered cell
                    if new_pos in self.other_numbered_cells:
                        return 0  # Cannot be adjacent to another numbered cell
                    
                    # Add to adjacent if in positions we care about
                    if new_pos in self.positions:
                        adjacent.add(new_pos)
        
        # Check if adjacent cells are all water (value=0)
        for pos in adjacent:
            var = self.pos_to_var.get(pos)
            if var and assignment.get(var, None) != 0:  # Not water
                return 0
        
        return 1
    
    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution.
        
        For partial assignments, we check:
        1. If there's already an island of the correct size, it must satisfy all island rules
        2. If the island is still growing, it must be possible to reach the required size
        3. Assigned cells adjacent to the island must be water (0)
        4. The island must not be adjacent to any other numbered cells
        """
        # Find the island containing the numbered cell (using only assigned variables)
        island = self.find_island(partial_assignment, self.numbered_cell_pos)
        
        # Check if island is already too large
        if len(island) > self.number_value:
            return 0
        
        # Check for adjacency to other numbered cells
        for pos in island:
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                new_pos = (nr, nc)
                
                if (0 <= nr < self.rows and 0 <= nc < self.cols and
                    new_pos not in island and
                    new_pos in self.other_numbered_cells):
                    return 0  # Cannot be adjacent to another numbered cell
        
        # If island has the correct size, all adjacent cells must be water
        if len(island) == self.number_value:
            # Check all cells adjacent to the island
            adjacent = set()
            for pos in island:
                row, col = pos
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    new_pos = (nr, nc)
                    
                    if (0 <= nr < self.rows and 0 <= nc < self.cols and
                        new_pos not in island and 
                        new_pos in self.positions):
                        adjacent.add(new_pos)
            
            # Any assigned adjacent cells must be water (0)
            for pos in adjacent:
                var = self.pos_to_var.get(pos)
                if var in partial_assignment and partial_assignment[var] != 0:
                    return 0
        
        # If island is too small, check if it could potentially grow to the right size
        elif len(island) < self.number_value:
            # Count unassigned cells adjacent to the current island
            potential_growth = 0
            frontier = set()
            
            for pos in island:
                row, col = pos
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    new_pos = (nr, nc)
                    
                    if (0 <= nr < self.rows and 0 <= nc < self.cols and
                        new_pos not in island and 
                        new_pos not in self.other_numbered_cells and
                        new_pos in self.positions):
                        var = self.pos_to_var.get(new_pos)
                        # Either unassigned or assigned as land (1)
                        if var not in partial_assignment or partial_assignment[var] == 1:
                            frontier.add(new_pos)
            
            # Check if the island could potentially grow to the required size
            if len(island) + len(frontier) < self.number_value:
                return 0
        
        return 1
    
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint.
        
        Each solution will include assignments for ALL variables passed to the constraint,
        ensuring consistent cardinality across solutions.
        """
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        
        # Get all unassigned variables from those passed to the constraint
        unassigned = self.get_unassigned()
        
        # If subset_vars is provided, only consider those variables
        if subset_vars is not None:
            subset_vars = set(subset_vars) & unassigned
            vars_to_assign = subset_vars
        else:
            vars_to_assign = unassigned
        
        # If nothing to assign, check if current assignment is valid
        if not vars_to_assign:
            if self.evaluate(assignment):
                yield assignment.copy()
            return
        
        # Get valid polyominoes that include the numbered cell
        valid_polyominoes = self.get_valid_polyominoes()
        
        # For each valid polyomino, generate a solution
        for poly in valid_polyominoes:
            # Start with the current assignment
            solution = assignment.copy()
            
            # Check if polyomino is consistent with current assignments
            valid = True
            for pos in poly:
                if pos == self.numbered_cell_pos:
                    continue
                
                var = self.pos_to_var.get(pos)
                if var is not None and var in solution and solution[var] != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check that polyomino is not adjacent to other numbered cells
            for pos in poly:
                row, col = pos
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    new_pos = (nr, nc)
                    
                    if (0 <= nr < self.rows and 0 <= nc < self.cols and
                        new_pos not in poly and
                        new_pos in self.other_numbered_cells):
                        valid = False
                        break
                
                if not valid:
                    break
                
            if not valid:
                continue
            
            # Get adjacent positions (must be water)
            adjacent = set()
            for pos in poly:
                row, col = pos
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    new_pos = (nr, nc)
                    
                    if (0 <= nr < self.rows and 0 <= nc < self.cols and
                        new_pos not in poly and 
                        new_pos in self.positions):
                        adjacent.add(new_pos)
            
            # Check if adjacent is consistent with current assignments
            for pos in adjacent:
                var = self.pos_to_var.get(pos)
                if var is not None and var in solution and solution[var] != 0:
                    valid = False
                    break
                
            if not valid:
                continue
            
            # Set polyomino cells to land (1)
            for pos in poly:
                if pos == self.numbered_cell_pos:
                    continue
                
                var = self.pos_to_var.get(pos)
                if var is not None and var in vars_to_assign:
                    solution[var] = 1
            
            # Set adjacent cells to water (0)
            for pos in adjacent:
                var = self.pos_to_var.get(pos)
                if var is not None and var in vars_to_assign:
                    solution[var] = 0
            
            # Identify remaining unassigned variables
            remaining = []
            for var in vars_to_assign:
                if var not in solution:
                    remaining.append(var)
            
            # Generate all combinations for remaining variables
            if not remaining:
                yield solution.copy()
            else:
                # Use binary counting method to generate all combinations
                for i in range(1 << len(remaining)):
                    full_solution = solution.copy()
                    for j, var in enumerate(remaining):
                        full_solution[var] = 1 if (i & (1 << j)) else 0
                    
                    # Final check using evaluate to ensure the solution is valid
                    if self.evaluate(full_solution):
                        yield full_solution.copy()
    
    def sample(self, partial_assignment=None, subset_vars=None):
        """Sample a random solution that satisfies the constraint."""
        solutions = list(self.possible_solutions(partial_assignment, subset_vars))
        if not solutions:
            return None
        return random.choice(solutions)



def create_grid(rows, cols, numbered_cells):
    """Create a Nurikabe grid with numbered cells.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        numbered_cells: List of (row, col, value) for numbered cells
        
    Returns:
        Grid where positive numbers are clues, 0 is empty
    """
    grid = np.zeros((rows, cols), dtype=int)
    for r, c, val in numbered_cells:
        grid[r, c] = val
    return grid

def print_grid(grid, highlight_cells = None, 
               highlight_adjacent = None, 
               solution_overlay = None):
    """Print a Nurikabe grid with optional highlighting.
    
    Args:
        grid: The Nurikabe grid
        highlight_cells: Set of (row, col) to highlight as part of a polyomino
        highlight_adjacent: Set of (row, col) to highlight as adjacent to a polyomino
        solution_overlay: Dictionary of {variable: value} assignments to display
    """
    rows, cols = grid.shape
    highlight_cells = highlight_cells or set()
    highlight_adjacent = highlight_adjacent or set()
    
    # Map from positions to values if solution_overlay is provided
    pos_values = {}
    if solution_overlay:
        for var, value in solution_overlay.items():
            pos_values[(var.row, var.col)] = value
    
    # Print top border
    print("+" + "---+" * cols)
    
    for r in range(rows):
        # Print cell contents
        row_str = "|"
        for c in range(cols):
            cell_val = grid[r, c]
            pos = (r, c)
            
            if cell_val > 0:  # Numbered cell
                row_str += f" {cell_val} |"
            elif pos in highlight_cells:  # Part of polyomino (white/island)
                row_str += " P |"
            elif pos in highlight_adjacent:  # Adjacent to polyomino (black/water)
                row_str += " A |"
            elif pos in pos_values:  # Solution value
                symbol = "■" if pos_values[pos] == 1 else "□"  # ■=land(1), □=water(0)
                row_str += f" {symbol} |"
            else:  # Empty cell
                row_str += "   |"
        
        print(row_str)
        
        # Print horizontal border
        print("+" + "---+" * cols)

class WaterPathConstraint(Constraint):
    """Constraint ensuring that all water cells (value=0) form a single connected component.
    
    This enforces the Nurikabe rule that all water cells must be connected.
    This constraint doesn't generate new solutions, but filters out invalid ones
    where water cells are disconnected.
    """
    
    def __init__(self, variables, grid_size, **kwargs):
        """Initialize a water path constraint.
        
        Args:
            variables: Set of variables in the grid
            grid_size: (rows, cols) of the grid
        """
        super().__init__(variables, 0, **kwargs)
        self.rows, self.cols = grid_size
        
        # Create position to variable mapping
        self.pos_to_var = {}
        for var in variables:
            self.pos_to_var[(var.row, var.col)] = var
    
    def __str__(self):
        return f"WaterPathConstraint: All water cells must form a single connected component"
    
    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        """Create a copy of this constraint."""
        return WaterPathConstraint(
            self.variables.copy(), 
            (self.rows, self.cols)
        )
    
    def size(self):
        return 2**len(self.get_unassigned())
    
    
    def find_connected_water(self, assignment):
        """Find all water cells connected to any water cell.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            Set of positions of water cells that form a single connected component
        """
        # Find all water cells
        water_cells = []
        for var, value in assignment.items():
            if value == 0:  # Water cell
                pos = (var.row, var.col)
                water_cells.append(pos)
        
        if not water_cells:
            return set()  # No water cells
        
        # Pick the first water cell and do a BFS to find all connected water
        visited = set()
        queue = [water_cells[0]]
        
        while queue:
            pos = queue.pop(0)
            if pos in visited:
                continue
                
            visited.add(pos)
            
            # Check orthogonal neighbors
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                new_pos = (nr, nc)
                
                # Skip if out of range or already visited
                if (not (0 <= nr < self.rows and 0 <= nc < self.cols) or 
                    new_pos in visited):
                    continue
                
                # Add to queue if it's a water cell (value=0)
                var = self.pos_to_var.get(new_pos)
                if var and var in assignment and assignment[var] == 0:
                    queue.append(new_pos)
        
        return visited
    
    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment.
        
        The constraint is satisfied if all water cells form a single connected component.
        """
        # Find all water cells
        water_cells = set()
        for var, value in assignment.items():
            if value == 0:  # Water cell
                pos = (var.row, var.col)
                water_cells.add(pos)
        
        if not water_cells:
            return 1  # No water cells, constraint is trivially satisfied
        
        # Find all connected water cells
        connected_water = self.find_connected_water(assignment)
        
        # Check if all water cells are connected
        return 1 if connected_water == water_cells else 0
    
    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution.
        
        For partial assignments, we check if all water cells assigned so far
        form a single connected component.
        """
        # Find all assigned water cells
        water_cells = set()
        for var, value in partial_assignment.items():
            if value == 0:  # Water cell
                pos = (var.row, var.col)
                water_cells.add(pos)
        
        if not water_cells:
            return 1  # No water cells, constraint is trivially satisfied
        
        # Find all connected water cells
        connected_water = self.find_connected_water(partial_assignment)
        
        # Check if all assigned water cells are in the connected component
        return 1 if connected_water == water_cells else 0
    
    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy."""
        # Get currently assigned variables
        assigned = self.get_assigned()
        if not assigned:
            return False  # No assignments, so no contradiction
        
        # Create a partial assignment from currently assigned variables
        partial_assignment = {var: var.value for var in assigned}
        
        # Use is_consistent to determine if there's a contradiction
        return self.is_consistent(partial_assignment) == 0
    
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint.
        
        This constraint doesn't generate new solutions, but filters existing ones.
        - If partial_assignment is None, yield nothing
        - If partial_assignment is provided, check if water forms a connected path
        
        Args:
            partial_assignment: Dictionary of assignments so far
            subset_vars: Subset of variables to generate assignments for (ignored)
        
        Yields:
            The original assignment if it satisfies the constraint, nothing otherwise
        """
        # If no partial assignment, return without yielding anything
        if partial_assignment is None:
            return
        
        # Check if the partial assignment is consistent with the constraint
        if self.is_consistent(partial_assignment):
            # If it's consistent, yield it as is
            yield partial_assignment.copy()

class IslandConnectedConstraint(Constraint):
    """Constraint ensuring that every land cell (value=1) belongs to an island connected to a numbered cell.
    
    This enforces the Nurikabe rule that all land cells must form islands that contain
    exactly one numbered cell each. It prevents "dangling islands" that are not
    connected to any numbered cell.
    
    This constraint doesn't generate new solutions, but filters out invalid ones
    where islands are disconnected from numbered cells.
    """
    
    def __init__(self, variables, numbered_cells, grid_size, **kwargs):
        """Initialize an island connectivity constraint.
        
        Args:
            variables: Set of variables representing non-numbered cells
            numbered_cells: List of (row, col) positions of numbered cells
            grid_size: (rows, cols) of the grid
        """
        super().__init__(variables, 1, **kwargs)
        self.numbered_cells = set(numbered_cells)
        self.rows, self.cols = grid_size
        
        # Create position to variable mapping
        self.pos_to_var = {}
        for var in variables:
            self.pos_to_var[(var.row, var.col)] = var
    
    def __str__(self):
        return f"IslandConnectedConstraint: All land cells must be part of islands connected to numbered cells"
    
    def __repr__(self):
        return self.__str__()
    
    def size(self):
        return 2**len(self.get_unassigned())
    
    def copy(self):
        """Create a copy of this constraint."""
        return IslandConnectedConstraint(
            self.variables.copy(),
            self.numbered_cells,
            (self.rows, self.cols)
        )
    
    def find_all_islands(self, assignment):
        """Find all separate islands in the grid.
        
        Args:
            assignment: Dictionary mapping variables to values
            
        Returns:
            List of sets, where each set contains positions of cells in an island
        """
        # Find all land cells
        land_cells = set()
        for var, value in assignment.items():
            if value == 1:  # Land cell
                pos = (var.row, var.col)
                land_cells.add(pos)
        
        # Also add numbered cells, which are always land
        all_land = land_cells | self.numbered_cells
        
        # Use BFS to find all distinct islands
        islands = []
        unvisited = all_land.copy()
        
        while unvisited:
            # Start a new island
            start = next(iter(unvisited))
            island = set()
            queue = [start]
            
            # BFS to find all cells in this island
            while queue:
                pos = queue.pop(0)
                if pos in island:
                    continue
                
                island.add(pos)
                unvisited.discard(pos)
                
                # Check orthogonal neighbors
                row, col = pos
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    new_pos = (nr, nc)
                    
                    # Skip if out of range or already visited
                    if (not (0 <= nr < self.rows and 0 <= nc < self.cols) or 
                        new_pos in island):
                        continue
                    
                    # Add to queue if it's a land cell or numbered cell
                    if new_pos in all_land:
                        queue.append(new_pos)
            
            # Add the island to our list
            islands.append(island)
        
        return islands
    
    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment.
        
        The constraint is satisfied if every island contains at least one numbered cell.
        """
        # Find all distinct islands
        islands = self.find_all_islands(assignment)
        
        # Check if each island contains at least one numbered cell
        for island in islands:
            if not island & self.numbered_cells:
                return 0  # Found an island with no numbered cell
        
        return 1  # All islands contain at least one numbered cell
    
    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution.
        
        For partial assignments, we check if all islands found so far
        either contain a numbered cell or could potentially connect to one.
        """
        # Find all distinct islands
        islands = self.find_all_islands(partial_assignment)
        
        # Check if each island contains at least one numbered cell
        for island in islands:
            if not island & self.numbered_cells:
                # No numbered cell in this island
                # Check if it could potentially connect to a numbered cell
                if not self.could_connect_to_numbered_cell(partial_assignment, island):
                    return 0  # Island can't connect to any numbered cell
        
        return 1  # All islands are consistent
    
    def could_connect_to_numbered_cell(self, partial_assignment, island):
        """Check if an island could potentially connect to a numbered cell.
        
        Args:
            partial_assignment: Dictionary mapping variables to values
            island: Set of positions forming the island
            
        Returns:
            True if the island could connect to a numbered cell, False otherwise
        """
        # Get all positions adjacent to the island
        frontier = set()
        for pos in island:
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                new_pos = (nr, nc)
                
                # Skip if out of range or already in the island
                if (not (0 <= nr < self.rows and 0 <= nc < self.cols) or 
                    new_pos in island):
                    continue
                
                # Add to frontier if it's a numbered cell or an unassigned cell
                if new_pos in self.numbered_cells:
                    return True  # Can connect directly to a numbered cell
                elif new_pos in self.pos_to_var:
                    var = self.pos_to_var[new_pos]
                    if var not in partial_assignment:
                        frontier.add(new_pos)  # Unassigned cell
        
        # If frontier is empty, island can't grow
        if not frontier:
            return False
        
        # Now check if any cell in the frontier could reach a numbered cell
        # through other unassigned cells
        for pos in frontier:
            if self.has_path_to_numbered_cell(partial_assignment, pos, island):
                return True
        
        return False
    
    def has_path_to_numbered_cell(self, partial_assignment, start_pos, avoid_positions):
        """Check if there's a potential path from start_pos to any numbered cell.
        
        Args:
            partial_assignment: Dictionary mapping variables to values
            start_pos: Starting position (row, col)
            avoid_positions: Set of positions to avoid (current island)
            
        Returns:
            True if a potential path exists, False otherwise
        """
        # BFS to find potential path to any numbered cell
        visited = set(avoid_positions)  # Start with current island as visited
        queue = [start_pos]
        
        while queue:
            pos = queue.pop(0)
            if pos in visited:
                continue
                
            visited.add(pos)
            
            # If we found a numbered cell, return True
            if pos in self.numbered_cells:
                return True
            
            # Check orthogonal neighbors
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                new_pos = (nr, nc)
                
                # Skip if out of range or already visited
                if (not (0 <= nr < self.rows and 0 <= nc < self.cols) or 
                    new_pos in visited):
                    continue
                
                # Add to queue if it's a numbered cell or an unassigned cell
                if new_pos in self.numbered_cells:
                    return True
                elif new_pos in self.pos_to_var:
                    var = self.pos_to_var[new_pos]
                    if var not in partial_assignment or partial_assignment[var] == 1:
                        queue.append(new_pos)
        
        # If we didn't find a numbered cell, return False
        return False
    
    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy."""
        # Get currently assigned variables
        assigned = self.get_assigned()
        if not assigned:
            return False  # No assignments, so no contradiction
        
        # Create a partial assignment from currently assigned variables
        partial_assignment = {var: var.value for var in assigned}
        
        # Use is_consistent to determine if there's a contradiction
        return self.is_consistent(partial_assignment) == 0
    
    def possible_solutions(self, partial_assignment=None, subset_vars=None):
        """Generate all possible solutions that satisfy the constraint.
        
        This constraint doesn't generate new solutions, but filters existing ones.
        - If partial_assignment is None, yield nothing
        - If partial_assignment is provided, check if all land cells are part of
          islands that contain numbered cells
        
        Args:
            partial_assignment: Dictionary of assignments so far
            subset_vars: Subset of variables to generate assignments for (ignored)
        
        Yields:
            The original assignment if it satisfies the constraint, nothing otherwise
        """
        # If no partial assignment, return without yielding anything
        if partial_assignment is None:
            return
        
        # Check if the partial assignment is consistent with the constraint
        if self.is_consistent(partial_assignment):
            # If it's consistent, yield it as is
            yield partial_assignment.copy()

#if __name__ == "__main__":

 
    # n_rows, n_cols = 4, 4
    # numbered_cells = [(0, 0, 2), (0, 2, 2)]
    # numbered_positions = [(r, c) for r, c, _ in numbered_cells]
    # grid = create_grid(n_rows, n_cols, numbered_cells)
    # variables = []
    # for i in range(n_rows):

    #     for j in range(n_cols):
    #         if (i, j) not in numbered_positions:
    #             var = Variable(f"v_{i}_{j}", domain={0, 1}, row=i, col=j)
    #             variables.append(var)

    # constraints = []    
    # integrated = []
    # for r, c, val in numbered_cells:
    #     island_constraint = PolyominoConstraint(
    #         set(variables), 
    #         (r, c), 
    #         val, 
    #         (n_rows, n_cols),
    #         all_numbered_cells=numbered_positions
    #     )
    #     constraints.append(island_constraint)

    #     print(island_constraint)
    #     assignments = list(island_constraint.possible_solutions())
    #     print_assignments(assignments)
        
    
    #     integrated = integrate_new_constraint(integrated, island_constraint)
    #     if not integrated:
    #         print("No solution found")
    #         break

    # if integrated is not None:
    #     for assignment in integrated:
    #         print_grid(grid, solution_overlay=assignment)

