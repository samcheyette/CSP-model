import sys
import os
# Ensure parent directory (CSP_working_model) is on sys.path for local imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from models.CSP_working_model.grammar import *
from typing import List, Tuple, Set
from agent import Agent
from constraints import EqualityConstraint, InequalityConstraint
from models.CSP_working_model.constraints import sort_constraints_by_relatedness, break_up_constraints
from models.CSP_working_model.utils.assignment_utils import (
    integrate_new_constraint,
    get_solved_variables,
    integrate_constraints,
)
import numpy as np
from models.CSP_working_model.utils.utils import *
import json
#from make_stimuli_mousetrack import make_stimulus
#from depth_predictor import DepthPredictor
from itertools import combinations, product

def get_ray_coords(board, row, col):
    rows, cols = board.shape
    rays = []
    
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    for dr, dc in directions:
        ray = []
        r, c = row + dr, col + dc
        
        while 0 <= r < rows and 0 <= c < cols and board[r, c] < 0 and board[r, c] != -2:  # While within bounds and not a black cell
            ray.append((r, c))
            r += dr
            c += dc
            
        rays.append(ray)
        
    return rays

def get_adjacent_coords(board, row, col):
    """Get coordinates of adjacent cells (up, right, down, left)."""
    rows, cols = board.shape
    adjacent = []
    
    # Directions: up, right, down, left
    for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < cols:
            adjacent.append((r, c))
            
    return adjacent

def get_illuminated_cells(board, lights):
    """Get all cells illuminated by the given light positions."""
    rows, cols = board.shape
    illuminated = set()
    
    # Add all light positions to illuminated cells
    for light_r, light_c in lights:
        illuminated.add((light_r, light_c))
        
        # Illuminate in all four directions until hitting a black cell or edge
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # Up, right, down, left
            r, c = light_r, light_c
            
            # Start from the light and move in each direction
            while True:
                r += dr
                c += dc
                
                # Check if we hit the edge or a black cell
                if r < 0 or r >= rows or c < 0 or c >= cols or board[r, c] == -2 or board[r, c] >= 0:
                    break
                    
                # Add this cell to illuminated set
                illuminated.add((r, c))
    
    return illuminated

def board_to_constraints(board):
    """Convert Light Up board to a list of constraints.
    
    Types of constraints:
    1. Numbered black cells: Exactly N adjacent cells must contain lights
    2. No overlapping lights: For each line of sight, at most one light
    3. All cells illuminated: Each white cell must be illuminated by at least one light
    """
    rows, cols = board.shape
    constraints = []
    
    # Create variables for all empty white cells
    variables = {}
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1 or board[r, c] == -3 or board[r, c] == -4:  # Empty white cell or marked
                variables[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", domain={0, 1}, row=r, col=c)
    
    # Add constraints for numbered black cells
    for r in range(rows):
        for c in range(cols):
            if 0 <= board[r, c] <= 4:  # Black cell with a number
                adjacent = get_adjacent_coords(board, r, c)
                adj_vars = [variables[f"v_{adj_r}_{adj_c}"] for adj_r, adj_c in adjacent 
                           if f"v_{adj_r}_{adj_c}" in variables]
                
                if adj_vars:  # If there are any adjacent variables
                    # Number of adjacent lights must equal the number in the cell
                    constraints.append(EqualityConstraint(set(adj_vars), int(board[r, c]), row=r, col=c))
    
    # Find all lines of sight (rays) and create constraints for no overlapping lights
    for r in range(rows):
        for c in range(cols):
            if f"v_{r}_{c}" in variables:
                rays = get_ray_coords(board, r, c)
                # For each ray (including the cell itself), create a constraint
                for ray in rays:
                    ray_vars = [variables[f"v_{ray_r}_{ray_c}"] for ray_r, ray_c in ray 
                               if f"v_{ray_r}_{ray_c}" in variables]
                    
                    # Add the cell itself to complete the line of sight
                    line_vars = [variables[f"v_{r}_{c}"]] + ray_vars
                    
                    if len(line_vars) > 1:  # Only create constraint if there are multiple variables
                        # At most one light in this line of sight (sum < 2)
                        constraints.append(InequalityConstraint(set(line_vars), 2, greater_than=False))
    
    # Add constraints for all cells being illuminated
    for r in range(rows):
        for c in range(cols):
            cell_key = f"v_{r}_{c}"
            if cell_key in variables:
                # This cell is illuminated if it contains a light or is in a ray from another light
                illuminators = []
                
                # Add the cell itself (can illuminate itself)
                illuminators.append(variables[cell_key])
                
                # Add cells that can illuminate this cell
                for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # Up, right, down, left
                    dr, dc = direction
                    ray_r, ray_c = r + dr, c + dc
                    
                    # Follow the ray in this direction
                    while 0 <= ray_r < rows and 0 <= ray_c < cols and board[ray_r, ray_c] < 0 and board[ray_r, ray_c] != -2:
                        ray_key = f"v_{ray_r}_{ray_c}"
                        if ray_key in variables:
                            illuminators.append(variables[ray_key])
                        ray_r += dr
                        ray_c += dc

                constraints.append(InequalityConstraint(set(illuminators), 0, greater_than=True))
    return constraints 



def board_to_partial_constraints(board, max_constraint_size = 2, 
                                subset_size = 2, coverage_probability = 1):

    full_constraints = board_to_constraints(board)
    
    return break_up_constraints(
        full_constraints, 
        max_constraint_size=max_constraint_size,
        subset_size=subset_size,
        coverage_probability=coverage_probability,
    )

def hash_board_state(board, row_clues=None, col_clues=None):
    """
    Create a compressed hash representation of a tents game state.
    Includes board state and row/column constraints.
    """
    import hashlib

    flat_board = board.flatten()
    board_str = f"{board.shape[0]}x{board.shape[1]}:" + ",".join(map(str, flat_board))
    
    # Include row and column clues in the hash since they're part of the game definition
    row_str, col_str = "", ""
    if row_clues is not None:
        row_str = "R:" + ",".join(map(str, row_clues))
    if col_clues is not None:
        col_str = "C:" + ",".join(map(str, col_clues))
    
    full_str = board_str + "|" + row_str + "|" + col_str
    hash_obj = hashlib.sha256(full_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]

class Game:    
    def __init__(self, board):

        self.initial_board = board.copy()
        self.board = board.copy()
        self.rows, self.cols = board.shape
    
    def reset(self):
        """Reset the board to its initial state."""
        self.board = self.initial_board.copy()
    
    def is_solved(self) -> bool:
        """Check if the puzzle is solved correctly."""
        # Find all light positions
        lights = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                 if self.board[r, c] == -3]  # -3 represents a light
        
        # Check numbered black cell constraints
        for r in range(self.rows):
            for c in range(self.cols):
                if 0 <= self.initial_board[r, c] <= 4:  # Numbered black cell
                    # Count adjacent lights
                    adjacent = get_adjacent_coords(self.initial_board, r, c)
                    adjacent_lights = sum(1 for adj_r, adj_c in adjacent 
                                         if self.board[adj_r, adj_c] == -3)
                    
                    # Check if number of adjacent lights matches constraint
                    if adjacent_lights != self.initial_board[r, c]:
                        return False
        
        # Check that no light illuminates another light
        for light_r, light_c in lights:
            # Check in all four directions
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                r, c = light_r + dr, light_c + dc
                # Continue in this direction until we hit a black cell or board edge
                while 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.board[r, c] == -2 or self.board[r, c] >= 0:  # Black cell
                        break
                    if self.board[r, c] == -3 and (r, c) != (light_r, light_c):  # Another light
                        return False
                    r += dr
                    c += dc
        
        # Calculate all illuminated cells
        illuminated = get_illuminated_cells(self.board, lights)
        
        # Check that all white cells are illuminated
        white_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                     if self.board[r, c] == -1 or self.board[r, c] == -3 or self.board[r, c] == -4]
        
        for r, c in white_cells:
            if (r, c) not in illuminated:
                return False
        
        return True
    
    def place_light(self, row, col):
        """Place a light at the specified position."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.initial_board[row, col] == -2 or self.initial_board[row, col] >= 0:  # Not a white cell
            raise ValueError(f"Cannot place light on non-white cell at ({row}, {col})")
        self.board[row, col] = -3  # -3 represents a light
    
    def mark_no_light(self, row, col):
        """Mark a cell as not containing a light."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.initial_board[row, col] == -2 or self.initial_board[row, col] >= 0:  # Not a white cell
            raise ValueError(f"Cannot mark non-white cell at ({row}, {col})")
        self.board[row, col] = -4  # -4 represents a "no light" mark
    
    def clear_cell(self, row, col):
        """Clear a light or mark from a cell."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.initial_board[row, col] == -2 or self.initial_board[row, col] >= 0:  # Not a white cell
            raise ValueError(f"Cannot clear non-white cell at ({row}, {col})")
        self.board[row, col] = -1  # Reset to empty white cell
    
    def get_board_state(self):
        """Return current board state."""
        return self.board.copy()
    

    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of all empty white cells."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols) 
                if self.board[r, c] == -1]
    
    def get_display_board(self):

        
        lights = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.board[r, c] == -3]
        illuminated = get_illuminated_cells(self.board, lights)
        rows, cols = self.board.shape

        new_board = np.zeros_like(self.board)
        for r in range(rows):
            for c in range(cols):
                if (r, c) in illuminated and self.board[r, c] != -3:
                    new_board[r, c] = -5  # Temporary value for illuminated cells
                else:
                    new_board[r, c] = self.board[r, c]
        
        return new_board
        
    def __str__(self):
        symbols = {
            -1: '·',  # Empty white cell
            -2: '■',  # Black cell (no number)
            0: '0',   # Black cell with number 0
            1: '1',   # Black cell with number 1
            2: '2',   # Black cell with number 2
            3: '3',   # Black cell with number 3
            4: '4',   # Black cell with number 4
            -3: 'L',  # Light bulb
            -4: 'X',  # Marked as "no light"
            -5: '○'   # Illuminated cell
        }
        display_board = self.get_display_board()
        s = ""
        for r in range(self.rows):
            s += ' '.join(symbols.get(int(display_board[r, c]), str(display_board[r, c])) for c in range(self.cols)) + "\n"
        return s
        
    
    def __repr__(self):
        return f"Game(\n{str(self)}\n)" 

def print_board(board):
    """Print a Light Up board."""
    symbols = {
        -1: '·',  # Empty white cell
        -2: '■',  # Black cell (no number)
        0: '0',   # Black cell with number 0
        1: '1',   # Black cell with number 1
        2: '2',   # Black cell with number 2
        3: '3',   # Black cell with number 3
        4: '4',   # Black cell with number 4
        -3: 'L',  # Light bulb
        -4: 'X'   # Marked as "no light"
    }

    lights = [(r, c) for r in range(board.shape[0]) for c in range(board.shape[1]) if board[r, c] == -3]
    illuminated = get_illuminated_cells(board, lights)
    rows, cols = board.shape

    new_board = np.zeros_like(board)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in illuminated and board[r, c] != -3:
                new_board[r, c] = -5  # Temporary value for illuminated
            else:
                new_board[r, c] = board[r, c]
    
    # Update symbols to include illuminated cells
    display_symbols = symbols.copy()
    display_symbols[-5] = '○'  # Illuminated cell
    
    for r in range(rows):
        print(' '.join(display_symbols.get(int(new_board[r, c]), str(new_board[r, c])) for c in range(cols)))
    print()

def generate_random_solution(rows, cols, black_cell_ratio=0.25, numbered_black_cell_ratio=0.5):
    """Generate a random valid Light Up puzzle solution.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        black_cell_ratio: Ratio of black cells to total cells
        numbered_black_cell_ratio: Ratio of black cells that should show numbers
        
    Returns:
        board: Numpy array representing the solved puzzle
    """
    # Initialize empty board
    board = np.full((rows, cols), -1)  # All white cells
    
    # Place random black cells
    n_black_cells = int(rows * cols * black_cell_ratio)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    
    for r, c in cells[:n_black_cells]:
        board[r, c] = -2  # Black cell (no number)
    
    # Track illuminated cells and light positions
    lights = []
    illuminated = set()
    
    # Helper function to check if placing a light would illuminate other lights
    def is_valid_light(r, c):
        # Check if the cell is already illuminated or is a black cell
        if (r, c) in illuminated or board[r, c] == -2 or board[r, c] >= 0:
            return False
        
        # Check if placing a light here would illuminate any existing lights
        # Check in all four directions
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r, c
            while True:
                nr += dr
                nc += dc
                
                # Stop at board edge
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                    
                # Stop at black cell
                if board[nr, nc] == -2 or board[nr, nc] >= 0:
                    break
                    
                # If we would illuminate another light, not valid
                if (nr, nc) in lights:
                    return False
        
        return True
    
    # Place lights where valid
    white_cells = [(r, c) for r, c in cells if board[r, c] == -1]
    random.shuffle(white_cells)
    
    for r, c in white_cells:
        if is_valid_light(r, c):
            # Place a light
            lights.append((r, c))
            board[r, c] = -3  # Light bulb
            
            # Update illuminated cells
            illuminated.add((r, c))
            
            # Illuminate in four directions
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r, c
                while True:
                    nr += dr
                    nc += dc
                    
                    # Stop at board edge or black cell
                    if not (0 <= nr < rows and 0 <= nc < cols) or board[nr, nc] == -2 or board[nr, nc] >= 0:
                        break
                        
                    illuminated.add((nr, nc))
    
    # Mark all remaining white cells as "no light"
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:  # Empty white cell
                board[r, c] = -4  # No light
    
    # Calculate numbers for black cells based on adjacent lights
    black_cells = [(r, c) for r in range(rows) for c in range(cols) if board[r, c] == -2]
    # Randomly select which black cells will have numbers
    random.shuffle(black_cells)
    numbered_count = int(len(black_cells) * numbered_black_cell_ratio)
    
    for i, (r, c) in enumerate(black_cells):
        # Count adjacent lights
        adjacent = get_adjacent_coords(board, r, c)
        adjacent_lights = sum(1 for adj_r, adj_c in adjacent if board[adj_r, adj_c] == -3)
        
        # Only update black cells that should display numbers
        if i < numbered_count:
            board[r, c] = adjacent_lights
        # Others remain as -2 (black cell with no number)
    
    # Verify solution is valid
    game = Game(board)
    if not game.is_solved():
        # If not valid, try again with different ratio
        #new_ratio = black_cell_ratio * random.uniform(0.5, 2)
        #new_ratio = max(0.1, min(0.5, new_ratio)) 
        return None
        #return generate_random_solution(rows, cols, new_ratio, numbered_black_cell_ratio)
    
    return board

def create_puzzle_from_solution(solution_board):

    puzzle_board = solution_board.copy()
    
    # Remove all light placements (convert to empty white cells)
    puzzle_board[puzzle_board == -3] = -1  # Remove light bulbs
    puzzle_board[puzzle_board == -4] = -1  # Remove "no light" markers
    
    return puzzle_board

def _count_lightup_solutions(board, limit=2):
    rows, cols = board.shape

    # Build variables for white cells
    index_of = {}
    coords = []
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:
                index_of[(r, c)] = len(coords)
                coords.append((r, c))
    n_vars = len(coords)
    if n_vars == 0:
        return 1

    # Hard assignments for cells marked explicitly (if any)
    assign = [-1] * n_vars
    def set_hard(r, c, val):
        if (r, c) in index_of:
            vid = index_of[(r, c)]
            assign[vid] = val

    # Treat -3 as light, -4 as no light if present
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -3:
                set_hard(r, c, 1)
            elif board[r, c] == -4:
                set_hard(r, c, 0)

    # Equality constraints from numbered black cells (sum(adjacent)=number)
    eq_cons = []  # list of {"unassigned": set(vid), "remaining": int}
    var_to_eq = [[] for _ in range(n_vars)]
    for r in range(rows):
        for c in range(cols):
            if 0 <= board[r, c] <= 4:
                adj = get_adjacent_coords(board, r, c)
                vars_here = []
                rem = int(board[r, c])
                for ar, ac in adj:
                    if board[ar, ac] == -1:
                        vars_here.append(index_of[(ar, ac)])
                    elif board[ar, ac] == -3:
                        rem -= 1
                if rem < 0 or rem > len(vars_here):
                    return 0
                if vars_here:
                    cid = len(eq_cons)
                    eq_cons.append({"unassigned": set(vars_here), "remaining": rem})
                    for vid in vars_here:
                        var_to_eq[vid].append(cid)

    # LOS (at most one) constraints: for each white cell, include itself and its rays
    los_cons = []  # list of {"unassigned": set(vid), "has_one": bool}
    var_to_los = [[] for _ in range(n_vars)]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in index_of:
                vid_self = index_of[(r, c)]
                rays = get_ray_coords(board, r, c)
                for ray in rays:
                    ls = [index_of[(rr, cc)] for rr, cc in ray if (rr, cc) in index_of]
                    line_vars = [vid_self] + ls
                    if len(line_vars) > 1:
                        cid = len(los_cons)
                        los_cons.append({"unassigned": set(line_vars), "has_one": (assign[vid_self] == 1)})
                        for vid in line_vars:
                            var_to_los[vid].append(cid)

    # Illumination constraints (>=1) for each white cell: itself + rays in four directions
    illum_cons = []  # list of {"unassigned": set(vid), "has_one": bool}
    var_to_illum = [[] for _ in range(n_vars)]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in index_of:
                vids = [index_of[(r, c)]]
                for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] < 0:
                        if (rr, cc) in index_of:
                            vids.append(index_of[(rr, cc)])
                        rr += dr
                        cc += dc
                cid = len(illum_cons)
                has_one = any(assign[v] == 1 for v in vids)
                unassigned = {v for v in vids if assign[v] == -1}
                illum_cons.append({"unassigned": unassigned, "has_one": has_one})
                for vid in vids:
                    var_to_illum[vid].append(cid)

    trail = []
    def undo(tr):
        for item in reversed(tr):
            t = item[0]
            if t == "var":
                _, vid, prev = item
                assign[vid] = prev
            elif t == "eq":
                _, cid, vid, dec = item
                eq_cons[cid]["unassigned"].add(vid)
                if dec:
                    eq_cons[cid]["remaining"] += 1
            elif t == "los":
                _, cid, vid = item
                los_cons[cid]["unassigned"].add(vid)
            elif t == "los_has":
                _, cid = item
                los_cons[cid]["has_one"] = False
            elif t == "illum":
                _, cid, vid = item
                illum_cons[cid]["unassigned"].add(vid)
            elif t == "illum_has":
                _, cid = item
                illum_cons[cid]["has_one"] = False

    def apply_assign(vid, val, tr):
        prev = assign[vid]
        if prev != -1:
            return prev == val
        assign[vid] = val
        tr.append(("var", vid, prev))
        # Equality constraints
        for cid in var_to_eq[vid]:
            if vid in eq_cons[cid]["unassigned"]:
                eq_cons[cid]["unassigned"].remove(vid)
                dec = (val == 1)
                if dec:
                    eq_cons[cid]["remaining"] -= 1
                tr.append(("eq", cid, vid, dec))
                if eq_cons[cid]["remaining"] < 0 or eq_cons[cid]["remaining"] > len(eq_cons[cid]["unassigned"]):
                    return False
        # LOS constraints
        for cid in var_to_los[vid]:
            if vid in los_cons[cid]["unassigned"]:
                los_cons[cid]["unassigned"].remove(vid)
                tr.append(("los", cid, vid))
            if val == 1 and not los_cons[cid]["has_one"]:
                los_cons[cid]["has_one"] = True
                tr.append(("los_has", cid))
                # force others in this LOS to 0
                for other in list(los_cons[cid]["unassigned"]):
                    if assign[other] == -1:
                        tr2 = []
                        if not apply_assign(other, 0, tr2):
                            for _ in range(len(tr2)):
                                pass
                            return False
                        tr.extend(tr2)
        # Illum constraints
        for cid in var_to_illum[vid]:
            if vid in illum_cons[cid]["unassigned"]:
                illum_cons[cid]["unassigned"].remove(vid)
                tr.append(("illum", cid, vid))
            if val == 1 and not illum_cons[cid]["has_one"]:
                illum_cons[cid]["has_one"] = True
                tr.append(("illum_has", cid))
        return True

    def propagate(tr):
        changed = True
        while changed:
            changed = False
            # Equality
            for cid, cons in enumerate(eq_cons):
                rem = cons["remaining"]
                un = cons["unassigned"]
                if rem < 0 or rem > len(un):
                    return False
                if rem == 0 and un:
                    for vid in list(un):
                        if assign[vid] == -1:
                            tr2 = []
                            if not apply_assign(vid, 0, tr2):
                                return False
                            tr.extend(tr2)
                            changed = True
                elif rem == len(un) and un:
                    for vid in list(un):
                        if assign[vid] == -1:
                            tr2 = []
                            if not apply_assign(vid, 1, tr2):
                                return False
                            tr.extend(tr2)
                            changed = True
            # Illumination (>=1)
            for cid, cons in enumerate(illum_cons):
                if cons["has_one"]:
                    continue
                un = cons["unassigned"]
                if not un:
                    return False
                if len(un) == 1:
                    vid = next(iter(un))
                    if assign[vid] == -1:
                        tr2 = []
                        if not apply_assign(vid, 1, tr2):
                            return False
                        tr.extend(tr2)
                        changed = True
        return True

    def all_assigned():
        for v in assign:
            if v == -1:
                return False
        return True

    def choose_var():
        # pick unassigned with highest participation
        best = -1
        best_deg = -1
        for vid in range(n_vars):
            if assign[vid] == -1:
                deg = len(var_to_eq[vid]) + len(var_to_los[vid]) + len(var_to_illum[vid])
                if deg > best_deg:
                    best_deg = deg
                    best = vid
        return best

    solutions = 0
    def dfs():
        nonlocal solutions
        if solutions >= limit:
            return
        tr = []
        if not propagate(tr):
            undo(tr)
            return
        if all_assigned():
            solutions += 1
            undo(tr)
            return
        vid = choose_var()
        for val in (0, 1):
            tr2 = []
            if apply_assign(vid, val, tr2):
                dfs()
            undo(tr2)
            if solutions >= limit:
                break
        undo(tr)

    dfs()
    return solutions

def has_unique_solution(board, max_size=100000):
    return _count_lightup_solutions(board, limit=2) == 1

def get_solutions(board: np.ndarray):

    constraints = board_to_constraints(board)
    constraints = sort_constraints_by_relatedness(constraints)
    
    assignments = []
    for constraint in constraints:
        assignments = integrate_new_constraint(assignments, constraint)
        if assignments is None: 
            return None
        
    
    return assignments


def solve_board(board, max_size=100000, print_every=100):
    constraints = board_to_constraints(board)
    constraints = sort_constraints_by_relatedness(constraints)




    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    for i, constraint in enumerate(constraints):


        new_assignments = integrate_new_constraint(assignments, constraint, max_size=max_size)
        if new_assignments is None or len(new_assignments) == 0:
            return np.array(board)
        
        if len(new_assignments) > max_size:
            new_assignments = new_assignments[:max_size]

        assignments = new_assignments
        if i % print_every == 0:
            print(constraint)
            print(f"Run {i}/{len(constraints)} constraints, {len(assignments)} assignments")
            solved_variables = get_solved_variables(assignments)
            solved_board = np.array(board)
            for v in solved_variables:
                solved_board[v.row][v.col] = -3 if solved_variables[v] == 1 else -4
            print_board(solved_board)

        if assignments and len(assignments) == 1:
            solved_variables = get_solved_variables(assignments)
            if len(solved_variables) == len(variables):
                break

    solved_board = np.array(board)
    solved_variables = get_solved_variables(assignments)

    for v in solved_variables:
        solved_board[v.row][v.col] = -3 if solved_variables[v] == 1 else -4
    return solved_board


def get_related_constraint(variables):
    random_variable = random.choice(list(variables))
    neighbor_constraints = random_variable.get_constraints()
    if neighbor_constraints:
        return random.choice(list(neighbor_constraints))
    else:
        return None

def generate_lightup_puzzle(rows, cols, min_black_cell_ratio, 
                            max_black_cell_ratio, min_numbered_black_cell_ratio, 
                            max_numbered_black_cell_ratio, max_attempts=10):
    """Generate a Light Up puzzle with a unique solution.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        max_attempts: Maximum number of attempts to generate a valid puzzle
        

    Returns:
        puzzle_board: The puzzle board with black cells and numbers
        solution_board: The solution board with lights placed
    """
    for attempt in range(max_attempts):

        black_cell_ratio = np.random.uniform(min_black_cell_ratio, max_black_cell_ratio)
        numbered_black_cell_ratio = np.random.uniform(min_numbered_black_cell_ratio, max_numbered_black_cell_ratio)
        # Generate a random solution
        solution_board = generate_random_solution(rows, cols, black_cell_ratio, numbered_black_cell_ratio)
        
        if solution_board is None:
            continue

        # Create a puzzle by removing lights but keeping black cells
        puzzle_board = create_puzzle_from_solution(solution_board)
        
        # Check if the puzzle has a unique solution with just the black cells as hints
        if has_unique_solution(puzzle_board):
            return puzzle_board, solution_board
    
    print(f"Failed to generate a Light Up puzzle with unique solution after {max_attempts} attempts")
    return None, None

def get_board_state(agent, game_board):
    """Get the board state with the agent's current assignments.
    
    Args:
        agent: The CSP agent with variable assignments
        game_board: The initial game board
        
    Returns:
        Board with the agent's assignments filled in
    """
    board = np.array(game_board, dtype=int)
    
    for v in agent.variables:
        if v.is_assigned():
            if v.value == 1:  # Light bulb
                board[v.row, v.col] = -3
            else:  # No light
                board[v.row, v.col] = -4
    
    return board

def run_lightup_simulation(unsolved_board, solved_board, memory_capacity=10, max_steps=100,
                           R_init=1, delta_R=0, ILtol_init=np.inf, delta_IL=0, gamma=1.0, board_args={}):
    """Run a simulation of an agent solving a Light Up puzzle.
    
    Returns the error rate (fraction of incorrectly assigned cells).
    """
    # Build constraints from puzzle board
    constraints = board_to_constraints(unsolved_board, **board_args)

    # True assignments for evaluation (used by agent.main)
    variables = set().union(*[c.variables for c in constraints])
    mapped = {}
    for v in variables:
        if v.row is not None and v.col is not None:
            mapped[v] = int(solved_board[v.row][v.col])
    true_assignments = mapped

    # Run the new Agent loop using agent.main
    from run_agent import main as run_agent_main
    agent, _ = run_agent_main(
        constraints,
        true_assignments,
        memory_capacity=memory_capacity,
        R_init=R_init,
        delta_R=delta_R,
        ILtol_init=ILtol_init,
        delta_IL=delta_IL,
        gamma=gamma,
        max_steps=max_steps,
        print_output=False,
    )

    # Compute simple error rate vs solved_board
    errors = 0.0
    total_vars = len(agent.variables)
    if total_vars == 0:
        return 0.0

    agent_board = unsolved_board.copy()
    for v in agent.variables:
        if v.is_assigned():
            agent_board[v.row][v.col] = -3 if v.value == 1 else -4
            if v.value == 0 and solved_board[v.row][v.col] == -3:
                errors += 1
            elif v.value == 1 and solved_board[v.row][v.col] == -4:
                errors += 1
        else:
            errors += 0.5
    return errors / total_vars

def get_difficulty(unsolved_board, solved_board, n_simulations=10, memory_capacity=12, max_steps=250, R_init=1, delta_R=0, ILtol_init=np.inf, delta_IL=0, gamma=1.0):
    """Calculate the difficulty of a Light Up puzzle.
    
    Returns a value between 0 and 1, where higher values indicate more difficult puzzles.
    """
    error_rates = []
    for _ in range(n_simulations):
        error_rate = run_lightup_simulation(
            unsolved_board=unsolved_board, 
            solved_board=solved_board,
            memory_capacity=memory_capacity,
            max_steps=max_steps,
            R_init=R_init,
            delta_R=delta_R,
            ILtol_init=ILtol_init,
            delta_IL=delta_IL,
            gamma=gamma,
        )
        error_rates.append(error_rate)
    
    return np.mean(error_rates)

def save_boards(boards: List[dict], filename: str, filepath=None):
    """Save multiple Light Up boards to a single JSON file."""
    if filepath is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'lightup')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    
    with open(filepath, 'w') as f:
        json.dump(boards, f, indent=2)

def load_boards(filename: str, filepath=None):
    """Load multiple Light Up boards from a JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'lightup', f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    
    with open(filepath, 'r') as f:
        return json.load(f)
    


def generate_boards_by_difficulty(min_difficulty, max_difficulty, n_boards, rows, cols, n_priority = 100):
    """Generate Light Up boards within a specified difficulty range.
    
    Args:
        min_difficulty: Minimum difficulty score (0-1)
        max_difficulty: Maximum difficulty score (0-1)
        n_boards: Number of boards to generate
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        List of dictionaries with board data
    """
    print(f"Generating {n_boards} Light Up boards of size {rows}x{cols}")
    print(f"Requested difficulty range: {min_difficulty:.2f}-{max_difficulty:.2f}")
    
    boards = []
    n_attempts = 0
    max_attempts = max(5000, n_boards * 50)  # Allow more attempts for more boards
    
    while len(boards) < n_boards and n_attempts < max_attempts:
        if n_attempts % 10 == 0:
            print(f"Current progress: {len(boards)}/{n_boards} boards generated ({n_attempts} attempts)")
            print(f"Current difficulty range: {min_difficulty:.2f} to {max_difficulty:.2f}")
        
        n_attempts += 1
        if n_attempts % 1000 == 0:
            # Gradually expand the difficulty range if we're struggling to find boards
            min_difficulty *= 0.95
            max_difficulty *= 1.05
            min_difficulty = max(0.0, min_difficulty)
            max_difficulty = min(1.0, max_difficulty)
            print(f"Adjusting difficulty range to {min_difficulty:.2f} to {max_difficulty:.2f}")
        
        # Generate a puzzle
        
        puzzle_board, solution_board = generate_lightup_puzzle(rows, cols, 0.05, 0.4, 0.1,0.9)
        if puzzle_board is None or solution_board is None:
            continue
        
        # Calculate difficulty


        difficulty = get_difficulty(
            puzzle_board, 
            solution_board, 
            n_simulations=10,
            memory_capacity=12, 
            max_steps=200,
            R_init=1,
            delta_R=0,
            ILtol_init=np.inf,
            delta_IL=0,
            gamma=1.0
        )
        
        print(f"Generating board {len(boards)}/{n_boards} (difficulty: {min_difficulty:.2f} - {max_difficulty:.2f}) found: {difficulty:.2f}")
        
        if difficulty >= min_difficulty and difficulty <= max_difficulty:
            board_entry = {
                "rows": rows,
                "cols": cols,
                "id": hash_board_state(puzzle_board),
                "priority": 1*(len(boards) < n_priority),
                "game_state": puzzle_board.tolist(),
                "game_board": solution_board.tolist(),
                "n_lights": int(np.sum(solution_board == -3)),
                "difficulty": difficulty
            }
            boards.append(board_entry)
            print(f"Found board {len(boards)}/{n_boards} with difficulty {difficulty:.2f}")
            print_board(puzzle_board)
            print_board(solution_board)
        else:
            print(f"Skipping board with difficulty {difficulty:.2f}")
    
    boards = sorted(boards, key=lambda x: x["difficulty"])
    
    for i, board in enumerate(boards):
        board["idx"] = i
        print(f"Board {i+1} (difficulty: {board['difficulty']:.2f}):")
        print_board(np.array(board["game_state"]))
        print_board(np.array(board["game_board"]))
        print()
    
    print(f"\nGenerated {len(boards)} boards with difficulties from {boards[0]['difficulty']:.2f} to {boards[-1]['difficulty']:.2f}")
    
    return boards



    return solved_board
if __name__ == "__main__":
    # rows, cols = 9, 9
    # puzzle_board, solution_board = generate_lightup_puzzle(rows, cols, 0.05, 0.5,
    #                                                         max_attempts=1000)
    # print_board(puzzle_board)
    # print_board(solution_board)

    # solved = smarter_solver(puzzle_board, 100, max_steps_per_path = 25, samples_per_step = 10, max_vars = 3,
    #                    tau = 1, max_size=100000, print_every = 10)
    
    # print_board(solved)
    
    
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rows, cols =  5,5
    n_boards = 400
    difficulty = "expert"
    for_website = True

    
    if difficulty == "easy":
        min_difficulty = 0.0
        max_difficulty = 0.25
    elif difficulty == "hard":
        min_difficulty = 0.25
        max_difficulty = 1.0
    elif difficulty == "expert":
        min_difficulty = 0.3
        max_difficulty = 1.0
    else:
        min_difficulty = 0.0
        max_difficulty = 1.0

    if for_website:
        filepath = os.path.join(current_dir, 'puzzle_website')
        filename = f"lightup_{rows}x{cols}_{difficulty}"
    else:
        filepath = os.path.join(current_dir, 'saved_boards', 'lightup')
        filename = f"lightup_{rows}x{cols}"
    
    print(f"Generating {n_boards} Light Up boards of size {rows}x{cols}")
    print(f"Difficulty range: {min_difficulty:.2f}-{max_difficulty:.2f}")
    
    boards = generate_boards_by_difficulty(min_difficulty, max_difficulty, 
                                           n_boards, rows, cols)

    save_boards(boards, filename, filepath=filepath)
    print(f"\nSaved {len(boards)} boards to {filename}.json") 