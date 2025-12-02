import sys
import os
# Ensure parent directory (CSP_working_model) is on sys.path for local imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from grammar import *
from agent import Agent
from constraints import EqualityConstraint, InequalityConstraint, Constraint, sort_constraints_by_relatedness, break_up_constraints
from utils.assignment_utils import integrate_new_constraint, get_solved_variables, integrate_constraints, print_assignments
import numpy as np
from utils.utils import get_powerset
from typing import List, Tuple, Set
import json



"""
NOTE: for the game boards, we use the following:
0:     empty cell
1:     tree
-1:    unknown
-3:    tent placed
-4:    marked as no-tent

EXAMPLE board with row/column clues:
board = 
[[ 0  1  0  0]  # 1
 [ 0  0  1  0]  # 1
 [ 0  0  0  1]  # 1
 [ 1  0  0  0]] # 0
 # 1  1  0  1   column clues

row_clues = [1, 1, 1, 0]
col_clues = [1, 1, 0, 1]
"""

def get_adjacent_coords(rows: int, cols: int, row: int, col: int, diagonal: bool = False) -> List[Tuple[int, int]]:
    """Get coordinates of adjacent squares"""
    adjacent = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_r, new_c = row + dr, col + dc
        if 0 <= new_r < rows and 0 <= new_c < cols:
            adjacent.append((new_r, new_c))
            
    if diagonal:
        for dr, dc in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_r, new_c = row + dr, col + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                adjacent.append((new_r, new_c))
                
    return adjacent

def get_all_windows(rows: int, cols: int) -> List[List[Tuple[int, int]]]:
    """Generate all 3x3 windows in the grid.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        
    Returns:
        List of windows, where each window is a list of coordinates
    """
    windows = []
    for r in range(rows):
        for c in range(cols):
            # Create a 3x3 window centered at (r, c)
            window = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        window.append((nr, nc))
            windows.append(window)
    return windows

def board_to_constraints(board, row_clues, col_clues):
    """Convert board state to list of constraints.
    
    Args:
        board: The current board state
        row_clues: List of required tents in each row
        col_clues: List of required tents in each column
        
    Board values:
    1: tree
    -1: unknown/empty cell
    -3: tent placed
    -4: marked as no-tent
    """
    rows, cols = board.shape
    constraints = []
    
    # Create tent variables for all non-tree cells
    tent_vars = {}
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 1:  # Not a tree
                tent_vars[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", row=r, col=c)
                adjacent = get_adjacent_coords(rows, cols, r, c, diagonal=False)
                # If not adjacent to any tree, must be 0
                if not any(board[adj_r, adj_c] == 1 for (adj_r, adj_c) in adjacent):
                    constraints.append(EqualityConstraint(set([tent_vars[f"v_{r}_{c}"]]), 0))

    # Add row/column sum constraints
    for r in range(rows):
        row_vars = [tent_vars[f"v_{r}_{c}"] for c in range(cols) if f"v_{r}_{c}" in tent_vars]
        if row_vars:
            constraints.append(EqualityConstraint(set(row_vars), row_clues[r], row=r, col=None))

    for c in range(cols):
        col_vars = [tent_vars[f"v_{r}_{c}"] for r in range(rows) if f"v_{r}_{c}" in tent_vars]
        if col_vars:
            constraints.append(EqualityConstraint(set(col_vars), col_clues[c], row=None, col=c))

    # Each tree must have an adjacent tent
    trees = [(r,c) for r in range(rows) for c in range(cols) if board[r,c] == 1]
    for r, c in trees:
        adjacent = get_adjacent_coords(rows, cols, r, c, diagonal=False)
        candidate_vars = [tent_vars[f"v_{adj_r}_{adj_c}"] for (adj_r, adj_c) in adjacent 
                         if f"v_{adj_r}_{adj_c}" in tent_vars]
        if candidate_vars:
            constraints.append(InequalityConstraint(set(candidate_vars), 0, greater_than=True, row=r, col=c))

    # No adjacent tents (only create each pair constraint once)
    for r in range(rows):
        for c in range(cols):
            key = f"v_{r}_{c}"
            if key in tent_vars:
                adjacent = get_adjacent_coords(rows, cols, r, c, diagonal=True)
                # Only consider cells with "higher" coordinates to avoid duplicates
                for adj_r, adj_c in adjacent:
                    if (adj_r > r) or (adj_r == r and adj_c > c):
                        adj_key = f"v_{adj_r}_{adj_c}"
                        if adj_key in tent_vars:
                            constraints.append(InequalityConstraint(
                                {tent_vars[key], tent_vars[adj_key]}, 
                                2, greater_than=False, row=r, col=c
                            ))



    return constraints



def board_to_partial_constraints(board, row_clues, col_clues, max_constraint_size = 4, 
                                subset_size = 3, coverage_probability = 1):

    full_constraints = board_to_constraints(board, row_clues, col_clues)
    
    return break_up_constraints(
        full_constraints, 
        max_constraint_size=max_constraint_size,
        subset_size=subset_size,
        coverage_probability=coverage_probability,
    )

def print_board(board: np.ndarray, row_clues: List[int], col_clues: List[int]):
    """Helper function to print a board with clues"""
    symbols = {
            -1: '?',  # unknown/empty
             1: '↟',   # tree
            -3: '⧍',  # tent
            -4: 'X'   # no tent
        }    
    # Print board with row clues
    for r in range(len(board)):
        row = [symbols[int(cell)] for cell in board[r]]
        print(' '.join(row), f" {row_clues[r]}")
    
    # Print column clues aligned with columns
    print(' '.join(str(clue) for clue in col_clues))
    print()

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
    """Represents a Tents puzzle game state"""
    def __init__(self, board: np.ndarray, row_clues: List[int], col_clues: List[int]):
        self.initial_board = board.copy()
        self.board = board.copy()
        self.row_clues = row_clues
        self.col_clues = col_clues
        self.rows, self.cols = board.shape
    
    def reset(self):
        """Reset the board to initial state"""
        self.board = self.initial_board.copy()
        
    def get_adjacent_coords(self, row: int, col: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        """Get coordinates of all valid adjacent squares"""
        return get_adjacent_coords(self.rows, self.cols, row, col, diagonal)
        
    def get_unrevealed_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unrevealed squares"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] in [-1, -3, -4]]
                
    def get_unmarked_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unmarked squares"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == -1]
                
    def place_tent(self, row: int, col: int):
        """Place a tent at the specified position"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] == 1:  # Can't place tent on a tree
            raise ValueError(f"Cannot place tent on a tree at ({row}, {col})")
        self.board[row, col] = -3
        
    def mark_no_tent(self, row: int, col: int):
        """Mark a cell as not containing a tent"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] == 1:  # Can't mark a tree
            raise ValueError(f"Cannot mark a tree at ({row}, {col})")
        self.board[row, col] = -4

    def unmark(self, row: int, col: int):
        """Remove tent or no-tent mark from a cell"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-3, -4]:  # Can only unmark tents and no-tent marks
            raise ValueError(f"Cannot unmark cell at ({row}, {col})")
        self.board[row, col] = -1
        
    def get_board_state(self):
        """Return current board state"""
        return self.board.copy()
    
    def is_solved(self):
        """Check if the game is solved"""
        # Check if all cells are marked
        if np.any(self.board == -1):
            return False
            
        # Check row clues
        for r in range(self.rows):
            if np.sum(self.board[r] == -3) != self.row_clues[r]:
                return False
                
        # Check column clues
        for c in range(self.cols):
            if np.sum(self.board[:, c] == -3) != self.col_clues[c]:
                return False
                
        # Check no adjacent tents
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == -3:  # Tent
                    adjacent = self.get_adjacent_coords(r, c, diagonal=True)
                    if any(self.board[adj_r, adj_c] == -3 for adj_r, adj_c in adjacent):
                        return False
        
        # Find all trees and tents
        trees = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.board[r,c] == 1]
        tents = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.board[r,c] == -3]
        
        if len(trees) != len(tents):
            return False
        
        # Build adjacency graph between trees and tents
        adjacency = {}  # (tree_r,tree_c) -> list of adjacent tent positions
        for tree_r, tree_c in trees:
            adjacent = self.get_adjacent_coords(tree_r, tree_c, diagonal=False)
            adjacency[(tree_r,tree_c)] = [
                (r,c) for (r,c) in adjacent 
                if (r,c) in tents
            ]
        
        # Try to find a complete matching using DFS
        def find_matching(tree_idx, used_tents, matching):
            if tree_idx == len(trees):
                return True
            
            tree = trees[tree_idx]
            for tent in adjacency[tree]:
                if tent not in used_tents:
                    used_tents.add(tent)
                    matching[tree] = tent
                    if find_matching(tree_idx + 1, used_tents, matching):
                        return True
                    used_tents.remove(tent)
                    matching.pop(tree)
            return False
        
        matching = {}
        used_tents = set()
        return find_matching(0, used_tents, matching)
    
    def __str__(self) -> str:
        """Return string representation of the board"""
        symbols = {
            -1: '?',  # unknown/empty
            1: '↟',   # tree
            -3: '⧍',  # tent
            -4: 'X'   # no tent
        }
        
        # Convert board to string representation
        rows = []
        for r in range(self.rows):
            row = [symbols.get(cell, str(cell)) for cell in self.board[r]]
            rows.append(' '.join(row) + f" {self.row_clues[r]}")
        
        # Add column clues at bottom
        rows.append(' '.join(str(clue) for clue in self.col_clues))
        
        return  '\n'.join(rows) + '\n'
        
    def __repr__(self) -> str:
        return f"Game(\n{str(self)}\n)" 

def get_board_state(agent, game_board):
    """Get the board state with the agent's current assignments.
    
    Args:
        agent: The CSP agent with variable assignments
        game_board: The initial game board (with -1 for unmarked squares)
        
    Returns:
        Board with the agent's assignments filled in
    """
    board = np.array(game_board, dtype=int)
    
    # Add the agent's assignments
    for v in agent.variables:
        if v.is_assigned():
            if v.value == 1:  # Tent
                board[v.row][v.col] = -3
            else:  # No tent
                board[v.row][v.col] = -4
    
    return board

def check_unique_solution(board: np.ndarray, row_clues: List[int], col_clues: List[int], max_size=2500):
    """Check if a tents puzzle has a unique solution (fast backtracking)."""
    is_unique = _count_tents_solutions(board, row_clues, col_clues, limit=2) == 1
    return is_unique, []

def _count_tents_solutions(board: np.ndarray, row_clues: List[int], col_clues: List[int], limit=2) -> int:
    rows, cols = board.shape

    # Variables: all non-tree cells (board != 1)
    index_of = {}
    coords = []
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 1:
                index_of[(r, c)] = len(coords)
                coords.append((r, c))
    n_vars = len(coords)
    if n_vars == 0:
        return 1

    assign = [-1] * n_vars  # -1 unknown, 0 no tent, 1 tent

    # Pre-assign cells that cannot be tents (not adjacent to any tree)
    def has_adjacent_tree(r, c):
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == 1:
                return True
        return False

    # Hard assignments if present
    for (r, c), vid in index_of.items():
        if board[r, c] == -3:
            assign[vid] = 1
        elif board[r, c] == -4:
            assign[vid] = 0
        elif not has_adjacent_tree(r, c):
            assign[vid] = 0

    # Row/Col constraints
    row_cons = []  # {unassigned:set, remaining:int}
    col_cons = []
    var_to_rows = [[] for _ in range(n_vars)]
    var_to_cols = [[] for _ in range(n_vars)]

    for r in range(rows):
        vars_here = []
        rem = int(row_clues[r])
        for c in range(cols):
            if (r, c) in index_of:
                vid = index_of[(r, c)]
                if assign[vid] == 1:
                    rem -= 1
                elif assign[vid] == -1:
                    vars_here.append(vid)
        if rem < 0 or rem > len(vars_here):
            return 0
        cid = len(row_cons)
        row_cons.append({"unassigned": set(vars_here), "remaining": rem})
        for vid in vars_here:
            var_to_rows[vid].append(cid)

    for c in range(cols):
        vars_here = []
        rem = int(col_clues[c])
        for r in range(rows):
            if (r, c) in index_of:
                vid = index_of[(r, c)]
                if assign[vid] == 1:
                    rem -= 1
                elif assign[vid] == -1:
                    vars_here.append(vid)
        if rem < 0 or rem > len(vars_here):
            return 0
        cid = len(col_cons)
        col_cons.append({"unassigned": set(vars_here), "remaining": rem})
        for vid in vars_here:
            var_to_cols[vid].append(cid)

    # Tree adjacency constraints (>=1)
    tree_cons = []  # {unassigned:set, has_one:bool}
    var_to_tree = [[] for _ in range(n_vars)]
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 1:
                neigh = []
                has_one = False
                for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in index_of:
                        vid = index_of[(nr, nc)]
                        if assign[vid] == 1:
                            has_one = True
                        elif assign[vid] == -1:
                            neigh.append(vid)
                if not has_one and not neigh:
                    return 0
                cid = len(tree_cons)
                tree_cons.append({"unassigned": set(neigh), "has_one": has_one})
                for vid in neigh:
                    var_to_tree[vid].append(cid)

    # No-adjacent-tents graph
    neighbors = [[] for _ in range(n_vars)]
    for (r, c), vid in index_of.items():
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in index_of:
                    neighbors[vid].append(index_of[(nr, nc)])

    trail = []
    def undo(tr):
        for item in reversed(tr):
            t = item[0]
            if t == "var":
                _, vid, prev = item
                assign[vid] = prev
            elif t == "row":
                _, cid, vid, dec = item
                row_cons[cid]["unassigned"].add(vid)
                if dec:
                    row_cons[cid]["remaining"] += 1
            elif t == "col":
                _, cid, vid, dec = item
                col_cons[cid]["unassigned"].add(vid)
                if dec:
                    col_cons[cid]["remaining"] += 1
            elif t == "tree":
                _, cid, vid = item
                tree_cons[cid]["unassigned"].add(vid)
            elif t == "tree_has":
                _, cid = item
                tree_cons[cid]["has_one"] = False

    def apply_assign(vid, val, tr):
        prev = assign[vid]
        if prev != -1:
            return prev == val
        assign[vid] = val
        tr.append(("var", vid, prev))

        # Row constraints
        for cid in var_to_rows[vid]:
            if vid in row_cons[cid]["unassigned"]:
                row_cons[cid]["unassigned"].remove(vid)
                dec = (val == 1)
                if dec:
                    row_cons[cid]["remaining"] -= 1
                tr.append(("row", cid, vid, dec))
                if row_cons[cid]["remaining"] < 0 or row_cons[cid]["remaining"] > len(row_cons[cid]["unassigned"]):
                    return False
        # Col constraints
        for cid in var_to_cols[vid]:
            if vid in col_cons[cid]["unassigned"]:
                col_cons[cid]["unassigned"].remove(vid)
                dec = (val == 1)
                if dec:
                    col_cons[cid]["remaining"] -= 1
                tr.append(("col", cid, vid, dec))
                if col_cons[cid]["remaining"] < 0 or col_cons[cid]["remaining"] > len(col_cons[cid]["unassigned"]):
                    return False
        # Tree constraints
        for cid in var_to_tree[vid]:
            if vid in tree_cons[cid]["unassigned"]:
                tree_cons[cid]["unassigned"].remove(vid)
                tr.append(("tree", cid, vid))
            if val == 1 and not tree_cons[cid]["has_one"]:
                tree_cons[cid]["has_one"] = True
                tr.append(("tree_has", cid))

        # Adjacency: if val==1, neighbors must be 0
        if val == 1:
            for nb in neighbors[vid]:
                if assign[nb] == 1:
                    return False
                if assign[nb] == -1:
                    tr2 = []
                    if not apply_assign(nb, 0, tr2):
                        return False
                    tr.extend(tr2)
        return True

    def propagate(tr):
        changed = True
        while changed:
            changed = False
            # Rows
            for cid, cons in enumerate(row_cons):
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
            # Cols
            for cid, cons in enumerate(col_cons):
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
            # Trees (>=1)
            for cid, cons in enumerate(tree_cons):
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
        # MRV-ish: pick unassigned with highest participation degree
        best = -1
        best_deg = -1
        for vid in range(n_vars):
            if assign[vid] == -1:
                deg = len(var_to_rows[vid]) + len(var_to_cols[vid]) + len(var_to_tree[vid]) + len(neighbors[vid])
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


def solve_board(board, row_clues, col_clues, max_size=None,
                print_every=None):
    # """Solve a Tents board.
    
    # Returns the solution board.
    # """
    # constraints = board_to_partial_constraints(board, row_clues, col_clues, 
    #                                            max_constraint_size=max_constraint_size, 
    #                                            subset_size=subset_size, 
    #                                            coverage_probability=coverage_probability)

    constraints = board_to_constraints(board, row_clues, col_clues)
    constraints = sort_constraints_by_relatedness(constraints)

    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    solved_board = np.array(board)


    for i, constraint in enumerate(constraints):
        #print(constraint)
        new_assignments = integrate_new_constraint(assignments, constraint, max_size=max_size)
        #print(new_assignments)
        n_assignments = len(new_assignments) if new_assignments else 0



        if new_assignments is None or len(new_assignments) == 0:
            return np.array(solved_board)
        elif max_size is not None and len(new_assignments) > max_size:
            new_assignments = new_assignments[:max_size]

        assignments = new_assignments

        
        if print_every is not None and i % print_every == 0:
            print(f"Run {i}/{len(constraints)}, {n_assignments} assignments, {constraint}")

            print_board(solved_board, row_clues, col_clues)
            solved_variables = get_solved_variables(assignments)
            for v in solved_variables:
                solved_board[v.row][v.col] = -3 if solved_variables[v] == 1 else -4

        if assignments and len(assignments) == 1:
            solved_variables = get_solved_variables(assignments)
            if len(solved_variables) == len(variables):
                break

    
    solved_board = np.array(board)
    solved_variables = get_solved_variables(assignments)

    for v in solved_variables:
        solved_board[v.row][v.col] = -3 if solved_variables[v] == 1 else -4

    #print_board(solved_board, row_clues, col_clues)
    return solved_board

def get_solutions(board, row_clues, col_clues):
    constraints = board_to_constraints(board, row_clues, col_clues)
    constraints = sort_constraints_by_relatedness(constraints)
    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    for constraint in constraints:
        assignments = integrate_new_constraint(assignments, constraint)
        solved_variables = get_solved_variables(assignments)
        if len(solved_variables) == len(variables):
            break
    return assignments

def try_generate_board(rows, cols, n_tents, max_placement_attempts=100):
    """Try to generate a valid board configuration.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        n_tents: Number of tents to place
        max_placement_attempts: Maximum number of attempts
        
    Returns:
        board: Board with trees placed
        tents: Set of tent positions
    """
    for _ in range(max_placement_attempts):
        board = np.full((rows, cols), -1, dtype=int)  # Initialize with -1 for unknown
        valid_tent_locations = np.ones((rows, cols), dtype=bool)
        tents = set()
        trees = set()
        
        # First place all tents (ensuring no adjacency)
        for _ in range(n_tents):
            # Find valid tent locations (not adjacent to other tents)
            valid_indices = np.argwhere(valid_tent_locations)
            if len(valid_indices) == 0:
                break
                
            # Place tent
            tent_r, tent_c = valid_indices[random.randint(0, len(valid_indices) - 1)]
            tents.add((tent_r, tent_c))
            
            # Update validity mask - no tents can be adjacent to this tent
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = tent_r + dr, tent_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        valid_tent_locations[nr, nc] = False
        
        # If we couldn't place all tents, try again
        if len(tents) < n_tents:
            continue
        
        # Then place trees next to tents
        success = True
        for tent_r, tent_c in tents:
            # Find valid tree positions adjacent to this tent
            adjacent = [(tent_r + dr, tent_c + dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
            valid_positions = [
                (r, c) for r, c in adjacent 
                if 0 <= r < rows and 0 <= c < cols and board[r, c] == -1
            ]
            
            if not valid_positions:
                success = False
                break
                
            # Place tree
            tree_r, tree_c = random.choice(valid_positions)
            board[tree_r, tree_c] = 1
            trees.add((tree_r, tree_c))
        
        if success:
            return board, tents
    
    return None, None

def generate_tents_puzzle(rows: int, cols: int, n_tents: int, max_attempts: int = 100,
                          avoid_trivial_counts: bool = False):
    """Generate a Tents puzzle with a unique solution.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        n_tents: Number of tents
        max_attempts: Maximum number of generation attempts
        
    Returns:
        puzzle_board: Board with trees placed
        solution_board: Board with complete solution
        row_clues: List of required tents in each row
        col_clues: List of required tents in each column
    """
    for _ in range(max_attempts):
        # Generate a random board with trees and tent positions
        board, tent_positions = try_generate_board(rows, cols, n_tents)
        
        if board is None:
            continue
        
        # Convert tent positions to a set of tuples
        tent_positions = {(r, c) for r, c in tent_positions}
        
        # Create solution board
        solution_board = board.copy()
        for r in range(rows):
            for c in range(cols):
                if solution_board[r, c] == -1:
                    if (r, c) in tent_positions:
                        solution_board[r, c] = -3  # Tent
                    else:
                        solution_board[r, c] = -4  # No tent
        
        # Calculate row and column clues
        row_clues = [sum(1 for c in range(cols) if (r, c) in tent_positions) for r in range(rows)]
        col_clues = [sum(1 for r in range(rows) if (r, c) in tent_positions) for c in range(cols)]

        # Optionally reject puzzles whose row/column counts are "trivial"
        # Define trivial heuristically as 0 or the maximal non-adjacent occupancy bound ~ ceil(L/2)
        if avoid_trivial_counts:
            max_row = int(np.ceil(cols / 2))
            max_col = int(np.ceil(rows / 2))
            if any((clue == 0 or clue == max_row) for clue in row_clues):
                continue
            if any((clue == 0 or clue == max_col) for clue in col_clues):
                continue
        
        # Create puzzle board (just trees)
        puzzle_board = board.copy()
        
        # Check if the puzzle has a unique solution
        is_unique, _ = check_unique_solution(puzzle_board, row_clues, col_clues)
        
        if is_unique:
            return puzzle_board, solution_board, row_clues, col_clues
    
    # If we failed after max_attempts, return None
    return None, None, None, None

def run_tents_simulation(max_steps, unsolved_board, solved_board, row_clues, col_clues, 
                       memory_capacity, R_init, delta_R, ILtol_init, delta_IL, gamma):
    """Run a simulation of an agent solving a Tents puzzle.
    
    Args:
        max_steps: Maximum number of steps for the agent to take
        unsolved_board: The initial puzzle board
        solved_board: The solution board
        row_clues: Required tents in each row
        col_clues: Required tents in each column
        memory_capacity: Agent's memory capacity
        search_budget: Agent's search budget
        beta_IL: Information loss parameter
        tau: Temperature parameter
    
    Returns:
        The error rate (fraction of incorrectly assigned cells).
    """
    constraints = board_to_constraints(unsolved_board, row_clues, col_clues)


    # Map true assignments from solved_board to avoid re-solving
    variables = set().union(*[c.variables for c in constraints])
    mapped = {}
    for v in variables:
        if v.row is not None and v.col is not None:
            mapped[v] = 1 if solved_board[v.row][v.col] == -3 else 0
    true_assignments = mapped

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
    
    errors = 0
    total_cells = 0
    for v in agent.variables:
        total_cells += 1
        if v.is_assigned():
            correct_value = 1 if solved_board[v.row][v.col] == -3 else 0
            if v.value != correct_value:
                errors += 1
        else:
            # If not assigned, count as partial error
            errors += 0.5
    
    return errors / total_cells if total_cells > 0 else 0

def get_difficulty(unsolved_board, solved_board, row_clues, col_clues, 
                      memory_capacity=10,
                      max_steps=100,
                      R_init=0.25,
                      delta_R=0,
                      ILtol_init=np.inf,
                      delta_IL=0,
                      gamma=1.0,
                      n_simulations=30):
    """Get the difficulty of a Tents puzzle.
    
    Args:
        unsolved_board: The initial puzzle board
        solved_board: The solution board
        row_clues: Required tents in each row
        col_clues: Required tents in each column
        n_simulations: Number of simulations to run
        memory_capacity: Agent's memory capacity
        search_budget: Agent's search budget
        
    Returns:
        The difficulty of the puzzle.
    """
    error_rates = []
    for _ in range(n_simulations):
        error_rate = run_tents_simulation(
            max_steps, unsolved_board, solved_board, row_clues, col_clues,
            memory_capacity, R_init, delta_R, ILtol_init, delta_IL, gamma
        )
        error_rates.append(error_rate)
    return np.mean(error_rates)

def generate_boards_by_difficulty(min_difficulty, max_difficulty, n_boards, rows, cols, min_tents, max_tents, n_priority = 100):
    """Generate Tents boards within a specified difficulty range.
    
    Args:
        min_difficulty: Minimum difficulty score (0-1)
        max_difficulty: Maximum difficulty score (0-1)
        n_boards: Number of boards to generate
        rows: Number of rows
        cols: Number of columns
        min_tents: Minimum number of tents
        max_tents: Maximum number of tents
        
    Returns:
        List of dictionaries with board data
    """
    boards = []
    n_attempts = 0
    while len(boards) < n_boards:
        if n_attempts % 10 == 0:
            print(f"Current difficulty range: {min_difficulty:.2f} to {max_difficulty:.2f}")
        n_attempts += 1
        if n_attempts % 500 == 0:
            min_difficulty *= 0.95
            max_difficulty *= 1.05
            print(f"Adjusting difficulty range to {min_difficulty:.2f} to {max_difficulty:.2f}")
            
        # Generate a puzzle with unique solution
        n_tents = np.random.randint(min_tents, max_tents+1)

        puzzle_board, solution_board, row_clues, col_clues = generate_tents_puzzle(
            rows, cols, n_tents, max_attempts=25, avoid_trivial_counts=min_difficulty > 0.3)
            
        if puzzle_board is None:
            print("Failed to generate a puzzle with a unique solution.")
            continue
            
        # Calculate difficulty
        difficulty = get_difficulty(
            puzzle_board, solution_board, row_clues, col_clues,
            memory_capacity=10,
            max_steps=100,
            R_init=0.25,
            delta_R=0,
            ILtol_init=np.inf,
            delta_IL=0,
            gamma=1.0,
            n_simulations=10
        )
            
        if difficulty >= min_difficulty and difficulty <= max_difficulty:
            boards.append({
                "rows": rows,
                "cols": cols,
                "n_tents": n_tents,
                "id": hash_board_state(puzzle_board, row_clues, col_clues),
                "priority": 1*(len(boards) < n_priority),
                "game_state": puzzle_board.tolist(),
                "game_board": solution_board.tolist(),
                "row_tent_counts": row_clues,
                "col_tent_counts": col_clues,
                "difficulty": difficulty
            })
            print(f"Found board {len(boards)}/{n_boards} with difficulty {difficulty:.2f}")
            print_board(puzzle_board, row_clues, col_clues)
            print_board(solution_board, row_clues, col_clues)
        else:
            print(f"Skipping board with difficulty {difficulty:.2f}")
    
    # Sort boards by difficulty and add indices
    boards = sorted(boards, key=lambda x: x["difficulty"])
    for i, board in enumerate(boards):
        board["idx"] = i
        print(f"Board {i+1} (difficulty: {board['difficulty']:.2f}):")
        print_board(np.array(board["game_state"]), board["row_tent_counts"], board["col_tent_counts"])
        print()
    
    return boards

def save_boards(boards: List[dict], filename: str, filepath=None):
    """Save multiple Tents boards to a single JSON file."""
    if filepath is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'tents')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(boards, f, indent=2)

def load_boards(filename: str, filepath=None):
    """Load multiple Tents boards from a JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'tents', f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    
    with open(filepath, 'r') as f:
        return json.load(f)
    


if __name__ == "__main__":


    current_dir = os.path.dirname(os.path.abspath(__file__))



    
    rows, cols = 9,9
    min_tents = rows * cols // 10
    max_tents = rows * cols // 4
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
        min_difficulty = 0.33
        max_difficulty = 1.0
    else:
        min_difficulty = 0.0
        max_difficulty = 1.0

    if for_website:
        filepath = os.path.join(current_dir, 'puzzle_website')
        if difficulty == "easy":
            filename = f"tents_{rows}x{cols}_easy"
        elif difficulty == "hard":
            filename = f"tents_{rows}x{cols}_hard"
        else:
            filename = f"tents_{rows}x{cols}_expert"
    else:
        filepath = os.path.join(current_dir, 'saved_boards', 'tents')
        filename = f"tents_{rows}x{cols}"
    
    print(f"Generating {n_boards} boards of size {rows}x{cols} with {min_tents} to {max_tents} tents...")
    
    boards = generate_boards_by_difficulty(
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        n_boards=n_boards,
        rows=rows,
        cols=cols,
        min_tents=min_tents,
        max_tents=max_tents
    )
    
    save_boards(boards, filename, filepath=filepath)
    print(f"\nSaved {len(boards)} boards to {filename}.json")