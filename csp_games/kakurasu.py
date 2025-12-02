import sys
import os
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from models.CSP_working_model.constraints import EqualityConstraint, InequalityConstraint, Constraint, sort_constraints_by_relatedness, break_up_constraints
from models.CSP_working_model.grammar import Variable
from models.CSP_working_model.special_constraints.kakurasu_constraints import WeightedSumConstraint
from models.CSP_working_model.utils.assignment_utils import integrate_new_constraint, get_solved_variables
import random
import json

def board_to_constraints(board: np.ndarray, row_sums: List[int], col_sums: List[int]) -> List[Constraint]:
    """Convert Kakurasu board state to list of constraints.
    
    Creates variables for each cell and sum constraints for:
    - Each row must sum to its target
    - Each column must sum to its target
    
    Board values:
    - -1: unmarked cell
    - 0: X marker (definitely not black)
    - 1: black cell
    
    Returns:
        List of constraints representing the Kakurasu puzzle
    """
    constraints = []
    rows, cols = board.shape
    
    # Create variables for each cell that's not marked as X (0)
    variables = {}
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 0:  # Not marked as X
                var_name = f"v_{r}_{c}"
                variables[var_name] = Variable(var_name, domain={0, 1}, row=r, col=c)
    
    # Create constraints for row sums
    for r in range(rows):
        # Get variables in this row
        row_vars = [variables[f"v_{r}_{c}"] for c in range(cols) 
                   if f"v_{r}_{c}" in variables]
        
        # Skip if no variables in this row
        if not row_vars:
            continue
            
        # Create weights for cells in this row (1-indexed)
        weights = {}
        for c in range(cols):
            var_name = f"v_{r}_{c}"
            if var_name in variables:
                weights[variables[var_name]] = c + 1  # 1-indexed weights
        
        # Create weighted sum constraint for this row
        row_constraint = WeightedSumConstraint(
            set(row_vars),  # Convert list to set to match the expected type
            weights=weights, 
            target=row_sums[r], 
            row=r, 
            col=None
        )
        constraints.append(row_constraint)
    
    # Create constraints for column sums
    for c in range(cols):
        # Get variables in this column
        col_vars = [variables[f"v_{r}_{c}"] for r in range(rows) 
                   if f"v_{r}_{c}" in variables]
        
        # Skip if no variables in this column
        if not col_vars:
            continue
            
        # Create weights for cells in this column (1-indexed)
        weights = {}
        for r in range(rows):
            var_name = f"v_{r}_{c}"
            if var_name in variables:
                weights[variables[var_name]] = r + 1  # 1-indexed weights
        
        # Create weighted sum constraint for this column
        col_constraint = WeightedSumConstraint(
            set(col_vars),  # Convert list to set to match the expected type
            weights=weights, 
            target=col_sums[c], 
            row=None, 
            col=c
        )
        constraints.append(col_constraint)
    
    return constraints

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
    """Class representing a Kakurasu game state."""
    
    def __init__(self, board: np.ndarray, row_sums: List[int], col_sums: List[int]):
        """Initialize game from a board and target sums.
        
        Args:
            board: rows x cols numpy array with:
                -1: unmarked cell
                0: X marker (definitely not black)
                1: black cell
            row_sums: Target sum for each row
            col_sums: Target sum for each column
        """
        self.rows, self.cols = board.shape
        self.board = board.copy()
        self.initial_board = board.copy()
        self.row_sums = row_sums.copy() if row_sums is not None else []
        self.col_sums = col_sums.copy() if col_sums is not None else []
    
    def unmarked_squares_remaining(self):
        """Return the number of unmarked squares on the board."""
        return np.sum(self.board == -1)
        
    def is_solved(self) -> bool:
        """Check if the game is solved - all rows and columns match their targets."""
        # Check if any unmarked cells remain
        if np.any(self.board == -1):
            return False
            
        # Check row sums
        for r in range(self.rows):
            row_sum = 0
            for c in range(self.cols):
                if self.board[r, c] == 1:  # Black cell
                    row_sum += c + 1  # 1-indexed position
            if row_sum != self.row_sums[r]:
                return False
                
        # Check column sums
        for c in range(self.cols):
            col_sum = 0
            for r in range(self.rows):
                if self.board[r, c] == 1:  # Black cell
                    col_sum += r + 1  # 1-indexed position
            if col_sum != self.col_sums[c]:
                return False
                
        return True
    
    def reset(self):
        """Reset the board to initial state."""
        self.board = self.initial_board.copy()
        
    def __str__(self) -> str:
        """Return string representation of the board."""
        result = []
        
        # Add column indices
        header = "   " + " ".join(f"{i+1:2d}" for i in range(self.cols)) + " |"
        result.append(header)
        
        # Add separator
        result.append("---" + "--" * self.cols + "-+")
        
        # Add board with row indices and sums
        for r in range(self.rows):
            row = f"{r+1:2d}|"
            for c in range(self.cols):
                cell = self.board[r, c]
                if cell == 1:  # Black cell
                    row += " ■"
                elif cell == 0:  # X marker
                    row += " ×"
                else:  # Unmarked
                    row += " ·"
            row += f" | {self.row_sums[r]}"
            result.append(row)
            
        # Add separator
        result.append("---" + "--" * self.cols + "-+")
        
        # Add column sums
        footer = "   " + " ".join(f"{s:2d}" for s in self.col_sums)
        result.append(footer)
        
        return "\n".join(result)
        
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unmarked cells."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == -1]
                
    def mark_black(self, row: int, col: int):
        """Mark a cell as black."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        self.board[row, col] = 1
        
    def mark_x(self, row: int, col: int):
        """Mark a cell with an X (not black)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        self.board[row, col] = 0
        
    def clear_cell(self, row: int, col: int):
        """Clear a cell back to unmarked."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        self.board[row, col] = -1
        
    def get_board_state(self):
        """Return current board state."""
        return self.board.copy()
        
    def get_value(self, row: int, col: int) -> int:
        """Get the value at the specified position."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        return self.board[row, col]


def solve_board(board, row_sums, col_sums, max_size=2500):
    """Solve a Kakurasu board.
    
    Returns the solution board.
    """
    constraints = board_to_constraints(board, row_sums, col_sums)
    constraints = sort_constraints_by_relatedness(constraints)
    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    for i, constraint in enumerate(constraints):
        new_assignments = integrate_new_constraint(assignments, constraint)
        if new_assignments is None or len(new_assignments) == 0:
            return np.array(board)
        elif len(new_assignments) > max_size:
            new_assignments = new_assignments[:max_size]

        assignments = new_assignments
        if assignments and len(assignments) == 1:
            solved_variables = get_solved_variables(assignments)
            if len(solved_variables) == len(variables):
                break

        print(f"Run {i} constraints, {len(assignments)} assignments")
    
    solved_board = np.array(board)
    solved_variables = get_solved_variables(assignments)

    for v in solved_variables:
        solved_board[v.row][v.col] = solved_variables[v]
    return solved_board



def get_board_state(agent, game_board):
    """Get the board state with the agent's current assignments.
    
    Args:
        agent: The CSP agent with variable assignments
        game_board: The initial game board (with -1 for unmarked squares)
        
    Returns:
        Board with the agent's assignments filled in
    """
    board = np.array(game_board, dtype=int) 
    
    for v in agent.variables:
        if v.is_assigned():
            board[v.row, v.col] = v.value
    
    return board

def print_board(board, row_sums, col_sums):
    """Print a Kakurasu board."""
    game = Game(board, row_sums, col_sums)
    print(game)
    print()


def load_boards(filename: str, filepath = None):
    """Load multiple Kakurasu boards from a JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'kakurasu', f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    
    with open(filepath, 'r') as f:
        return json.load(f)
    

def generate_random_kakurasu(rows, cols,n_black=None):
    """Generate a random valid Kakurasu puzzle.
    
    The algorithm creates a solution by randomly placing black cells,
    then calculates the row and column sums.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        min_black_cells: Minimum number of black cells (default: rows)
        max_black_cells: Maximum number of black cells (default: rows * cols // 2)
        
    Returns:
        board: Numpy array representing the solution
        row_sums: List of target sums for each row
        col_sums: List of target sums for each column
    """

    
    # Generate a random number of black cells
    if n_black is None:
        n_black = random.randint(rows, cols * rows // 2)
    
    # Create an empty board
    board = np.zeros((rows, cols), dtype=int)
    
    # Randomly place black cells
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    for r, c in cells[:n_black]:
        board[r, c] = 1
        
    # Calculate row and column sums
    row_sums = []
    for r in range(rows):
        row_sum = 0
        for c in range(cols):
            if board[r, c] == 1:
                row_sum += c + 1  # 1-indexed position
        row_sums.append(row_sum)
        
    col_sums = []
    for c in range(cols):
        col_sum = 0
        for r in range(rows):
            if board[r, c] == 1:
                col_sum += r + 1  # 1-indexed position
        col_sums.append(col_sum)
        
    return board, row_sums, col_sums

def generate_kakurasu_puzzle(rows: int, cols: int, max_attempts: int = 100,
                             min_black_ratio: float = 0.15, max_black_ratio: float = 0.85):
    """Generate a Kakurasu puzzle with a unique solution.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        min_black_cells: Minimum number of black cells
            max_black_cells: Maximum number of black cells
            max_attempts: Maximum number of attempts to generate a valid puzzle
        
    Returns:
        puzzle_board: The puzzle board with -1 for unmarked cells
        solution_board: The solution board with 0s and 1s
        row_sums: Target sums for each row
        col_sums: Target sums for each column
    """
    total_cells = rows * cols
    # Derive black cell bounds from ratios
    min_black_cells = int(np.ceil(min_black_ratio * total_cells))
    max_black_cells = int(np.floor(max_black_ratio * total_cells))
    # Ensure sane ordering
    min_black_cells = max(0, min_black_cells)
    max_black_cells = min(total_cells, max_black_cells)
    
    # Try different black cell counts within the range
    for black_cell_count in range(min_black_cells, max_black_cells + 1):
        for _ in range(max_attempts // (max_black_cells - min_black_cells + 1)):
            # Generate a random solution
            solution_board, row_sums, col_sums = generate_random_kakurasu(rows, cols, black_cell_count)

            # Enforce black/white ratio bounds
            black_ratio = float(np.sum(solution_board == 1)) / float(total_cells)
            if not (min_black_ratio <= black_ratio <= max_black_ratio):
                continue
            
            # Start with no cells revealed (only sums provided)
            puzzle_board = np.full((rows, cols), -1, dtype=int)

            if has_unique_solution(puzzle_board, row_sums, col_sums):
                return puzzle_board, solution_board, row_sums, col_sums

            # No prefill: return only if sums-alone puzzle is unique
    
    # If we still failed, give up and return None
    return None, None, None, None

def save_boards(boards: List[dict], filename: str, filepath = None):
    """Save multiple Kakurasu boards to a single JSON file."""
    if filepath is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'kakurasu')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    
    with open(filepath, 'w') as f:
        json.dump(boards, f, indent=2)



def run_kakurasu_simulation(max_steps, unsolved_board, solved_board, row_hints, col_hints, 
                          memory_capacity, R_init, delta_R, ILtol_init, delta_IL, gamma):
    constraints = board_to_constraints(unsolved_board, row_hints, col_hints)


    # Map true assignments from solved_board to avoid re-solving
    variables = set().union(*[c.variables for c in constraints])
    mapped = {}
    for v in variables:
        if v.row is not None and v.col is not None:
            mapped[v] = int(solved_board[v.row][v.col])
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
            correct_value = solved_board[v.row][v.col]
            if v.value != correct_value:
                errors += 1
        else:
            # If not assigned, count as partial error
            errors += 0.5
    
    return errors / total_cells if total_cells > 0 else 0

def get_difficulty(unsolved_board, solved_board, row_hints, col_hints, n_simulations=10, memory_capacity=10, R_init=0.25, delta_R=0, ILtol_init=np.inf, delta_IL=0, gamma=1.0):
    """Calculate the difficulty of a Kakurasu puzzle.
    
    Returns a value between 0 and 1, where higher values indicate more difficult puzzles.
    """
    error_rates = []
    for _ in range(n_simulations):
        error_rate = run_kakurasu_simulation(
            max_steps=100, 
            unsolved_board=unsolved_board, 
            solved_board=solved_board,
            row_hints=row_hints,
            col_hints=col_hints,
            memory_capacity=memory_capacity, 
            R_init=R_init,
            delta_R=delta_R,
            ILtol_init=ILtol_init,
            delta_IL=delta_IL,
            gamma=gamma
        )
        error_rates.append(error_rate)
    
    return np.mean(error_rates)




    

def has_unique_solution(board, row_sums, col_sums, max_size=2500):
    """Check uniqueness via fast MRV+propagation solution counter."""
    return _count_kakurasu_solutions(board, row_sums, col_sums, limit=2) == 1

def _count_kakurasu_solutions(board, row_sums, col_sums, limit=2):
    rows, cols = board.shape
    # Variables: cells where board != 0 (0 means forbidden X), domain {0,1}; 1 means black
    index_of = {}
    coords = []
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 0:
                index_of[(r, c)] = len(coords)
                coords.append((r, c))
    n_vars = len(coords)
    if n_vars == 0:
        # No variables, check if sums are all zero
        return 1 if (all(s == 0 for s in row_sums) and all(s == 0 for s in col_sums)) else 0

    # Precompute weights
    w_row = [0] * n_vars  # weight contribution to row sum (c+1)
    w_col = [0] * n_vars  # weight contribution to col sum (r+1)
    row_vars = [[] for _ in range(rows)]
    col_vars = [[] for _ in range(cols)]
    for vid, (r, c) in enumerate(coords):
        w_row[vid] = c + 1
        w_col[vid] = r + 1
        row_vars[r].append(vid)
        col_vars[c].append(vid)

    # Initial assignment from prefilled 1s
    assign = [-1] * n_vars
    row_rem = list(int(x) for x in row_sums)
    col_rem = list(int(x) for x in col_sums)
    row_sum_unassigned = [sum(w_row[v] for v in row_vars[r]) for r in range(rows)]
    col_sum_unassigned = [sum(w_col[v] for v in col_vars[c]) for c in range(cols)]

    for vid, (r, c) in enumerate(coords):
        if board[r, c] == 1:
            assign[vid] = 1
            row_rem[r] -= w_row[vid]
            col_rem[c] -= w_col[vid]
    # Remove prefilled vars from unassigned sums
    for r in range(rows):
        row_sum_unassigned[r] = sum(w_row[v] for v in row_vars[r] if assign[v] == -1)
    for c in range(cols):
        col_sum_unassigned[c] = sum(w_col[v] for v in col_vars[c] if assign[v] == -1)

    # Quick infeasibility checks
    for r in range(rows):
        if row_rem[r] < 0 or row_rem[r] > row_sum_unassigned[r]:
            return 0
    for c in range(cols):
        if col_rem[c] < 0 or col_rem[c] > col_sum_unassigned[c]:
            return 0

    def apply_assign(vid, val, trail):
        if assign[vid] != -1:
            return assign[vid] == val
        assign[vid] = val
        trail.append(("var", vid))
        r, c = coords[vid]
        # update unassigned sums
        row_sum_unassigned[r] -= w_row[vid]
        col_sum_unassigned[c] -= w_col[vid]
        trail.append(("row_un", r, w_row[vid]))
        trail.append(("col_un", c, w_col[vid]))
        if val == 1:
            row_rem[r] -= w_row[vid]
            col_rem[c] -= w_col[vid]
            trail.append(("row_rem", r, w_row[vid]))
            trail.append(("col_rem", c, w_col[vid]))
            if row_rem[r] < 0 or col_rem[c] < 0:
                return False
        # bounds check
        if row_rem[r] > row_sum_unassigned[r]:
            return False
        if col_rem[c] > col_sum_unassigned[c]:
            return False
        return True

    def undo(trail):
        for item in reversed(trail):
            t = item[0]
            if t == "var":
                _, vid = item
                assign[vid] = -1
            elif t == "row_un":
                _, r, w = item
                row_sum_unassigned[r] += w
            elif t == "col_un":
                _, c, w = item
                col_sum_unassigned[c] += w
            elif t == "row_rem":
                _, r, w = item
                row_rem[r] += w
            elif t == "col_rem":
                _, c, w = item
                col_rem[c] += w

    def propagate(trail):
        changed = True
        while changed:
            changed = False
            # Row forcing
            for r in range(rows):
                rem = row_rem[r]
                un_sum = row_sum_unassigned[r]
                if rem < 0 or rem > un_sum:
                    return False
                if rem == 0 and un_sum > 0:
                    # force all unassigned row vars to 0
                    for vid in list(row_vars[r]):
                        if assign[vid] == -1:
                            tr2 = []
                            if not apply_assign(vid, 0, tr2):
                                return False
                            trail.extend(tr2)
                            changed = True
                elif rem == un_sum and un_sum > 0:
                    # force all unassigned row vars to 1
                    for vid in list(row_vars[r]):
                        if assign[vid] == -1:
                            tr2 = []
                            if not apply_assign(vid, 1, tr2):
                                return False
                            trail.extend(tr2)
                            changed = True
            # Column forcing
            for c in range(cols):
                rem = col_rem[c]
                un_sum = col_sum_unassigned[c]
                if rem < 0 or rem > un_sum:
                    return False
                if rem == 0 and un_sum > 0:
                    for vid in list(col_vars[c]):
                        if assign[vid] == -1:
                            tr2 = []
                            if not apply_assign(vid, 0, tr2):
                                return False
                            trail.extend(tr2)
                            changed = True
                elif rem == un_sum and un_sum > 0:
                    for vid in list(col_vars[c]):
                        if assign[vid] == -1:
                            tr2 = []
                            if not apply_assign(vid, 1, tr2):
                                return False
                            trail.extend(tr2)
                            changed = True
        return True

    def all_assigned():
        for v in assign:
            if v == -1:
                return False
        return True

    def choose_var():
        # Pick unassigned var with largest weight sum to prune faster
        best = -1
        best_score = -1
        for vid in range(n_vars):
            if assign[vid] == -1:
                score = w_row[vid] + w_col[vid]
                if score > best_score:
                    best_score = score
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
            # Validate residuals are zero
            if all(x == 0 for x in row_rem) and all(x == 0 for x in col_rem):
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

def generate_boards_by_difficulty(min_difficulty: float, max_difficulty: float, n_boards: int, rows: int, cols: int, n_priority = 100,
                                  min_black_ratio: float = 0.1, max_black_ratio: float = 0.9) -> List[dict]:
    """Generate a specified number of Kakurasu boards within a given difficulty range.
    
    Args:
        min_difficulty: Minimum difficulty level for boards (0.0-1.0)
        max_difficulty: Maximum difficulty level for boards (0.0-1.0)
        n_boards: Number of boards to generate
        rows: Number of rows for the boards
        cols: Number of columns for the boards
        
    Returns:
        List of dictionaries containing board information
    """
    print(f"Generating {n_boards} Kakurasu boards of size {rows}x{cols}")
    print(f"Requested difficulty range: {min_difficulty:.2f}-{max_difficulty:.2f}")
    
    boards = []
    n_attempts = 0
    max_attempts = max(5000, n_boards * 50)  # Allow more attempts for more boards
    
    while len(boards) < n_boards and n_attempts < max_attempts:
        if n_attempts % 10 == 0:
            print(f"Current progress: {len(boards)}/{n_boards} boards generated ({n_attempts} attempts)")
            print(f"Current difficulty range: {min_difficulty:.2f} to {max_difficulty:.2f}")
        
        n_attempts += 1
        if n_attempts % 100 == 0:
            # Gradually expand the difficulty range if we're struggling to find boards
            min_difficulty *= 0.95
            max_difficulty *= 1.05
            min_difficulty = max(0.0, min_difficulty)
            max_difficulty = min(1.0, max_difficulty)
            print(f"Adjusting difficulty range to {min_difficulty:.2f} to {max_difficulty:.2f}")

        # Generate a puzzle
        puzzle_board, solution_board, row_sums, col_sums = generate_kakurasu_puzzle(rows, cols,
            min_black_ratio=min_black_ratio, max_black_ratio=max_black_ratio)
        if puzzle_board is None or solution_board is None or row_sums is None or col_sums is None:
            continue
            
        # Calculate difficulty
        difficulty = get_difficulty(
            puzzle_board, 
            solution_board, 
            row_sums, 
            col_sums, 
            n_simulations=10,
            memory_capacity=10, 
            R_init=0.25,
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
                "id": hash_board_state(puzzle_board, row_sums, col_sums),
                "priority": 1*(len(boards) < n_priority),
                "game_state": puzzle_board.tolist(),
                "game_board": solution_board.tolist(),
                "row_sums": row_sums,
                "col_sums": col_sums,
                "difficulty": difficulty
            }
            boards.append(board_entry)
            print(f"Found board {len(boards)}/{n_boards} with difficulty {difficulty:.2f}")
            print_board(puzzle_board, row_sums, col_sums)
            print_board(solution_board, row_sums, col_sums)
        else:
            print(f"Skipping board with difficulty {difficulty:.2f}")
    
    # Sort boards by difficulty
    boards = sorted(boards, key=lambda x: x["difficulty"])
    
    # Update indexes after sorting
    for i, board in enumerate(boards):
        board["idx"] = i
        print(f"Board {i+1} (difficulty: {board['difficulty']:.2f}):")
        print_board(np.array(board["game_state"]), board["row_sums"], board["col_sums"])
        print_board(np.array(board["game_board"]), board["row_sums"], board["col_sums"])
        print()
    
    print(f"\nGenerated {len(boards)} boards with difficulties from {boards[0]['difficulty']:.2f} to {boards[-1]['difficulty']:.2f}")
    
    return boards


if __name__ == "__main__":
    # Configuration for board generation

    current_dir = os.path.dirname(os.path.abspath(__file__))

    rows, cols = 7,7
    n_boards = 400  # Number of boards per difficulty level
    difficulty = "expert"
    for_website = True



    # Set difficulty ranges based on selected difficulty level
    if difficulty == "easy":
        min_difficulty = 0.0
        max_difficulty = 0.25
    elif difficulty == "hard":
        min_difficulty = 0.25
        max_difficulty = 1.0
    elif difficulty == "expert":
        min_difficulty = 0.35
        max_difficulty = 1.0
    else:
        min_difficulty = 0.0
        max_difficulty = 1.0

    # Set the filepath and filename based on the destination
    if for_website:
        filepath = os.path.join(current_dir, 'puzzle_website')
        filename = f"kakurasu_{rows}x{cols}_{difficulty}"
    else:
        filepath = os.path.join(current_dir, 'saved_boards', 'kakurasu')
        filename = f"kakurasu_{rows}x{cols}"
    
    print(f"Generating {n_boards} Kakurasu boards of size {rows}x{cols}")
    print(f"Difficulty range: {min_difficulty:.2f}-{max_difficulty:.2f}")

    
    # Generate boards within the specified difficulty range
    boards = generate_boards_by_difficulty(min_difficulty, max_difficulty, n_boards, rows, cols)

    print(boards)
    
    # # Save boards to the specified location
    save_boards(boards, filename, filepath=filepath)
    print(f"\nSaved {len(boards)} boards to {filename}.json")
    
