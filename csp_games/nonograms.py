import sys
import os
# Ensure parent directory (CSP_working_model) is on sys.path for local imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from models.CSP_working_model.grammar import *
from typing import List, Tuple, Set
from agent import Agent
from constraints import EqualityConstraint, InequalityConstraint, Constraint, sort_constraints_by_relatedness, break_up_constraints
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
from models.CSP_working_model.special_constraints.nonograms_constraints import RunLengthsConstraint


def board_to_constraints(board: np.ndarray, row_hints: List[List[int]], col_hints: List[List[int]]) -> List[Constraint]:
    """Convert Nonogram board state to list of constraints.
    
    Creates variables for each cell and run length constraints for:
    - Each row must match its run length hints
    - Each column must match its run length hints
    
    Board values:
    - -1: unmarked cell
    - 0: X marker (definitely not filled)
    - 1: filled (black) cell
    
    Args:
        board: The current board state
        row_hints: List of run length hints for each row
        col_hints: List of run length hints for each column
        
    Returns:
        List of constraints representing the Nonogram puzzle
    """
    constraints = []
    rows, cols = board.shape
    
    # Create variables for each cell that's not marked as X (0) or filled (1)
    variables = {}
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:  # Unmarked cell
                var_name = f"v_{r}_{c}"
                variables[var_name] = Variable(var_name, domain={0, 1}, row=r, col=c)
            elif board[r, c] == 1:  # Pre-filled black cell
                var_name = f"v_{r}_{c}"
                var = Variable(var_name, domain={0, 1}, row=r, col=c)
                var.assign(1)  # Set as filled
                variables[var_name] = var
            elif board[r, c] == 0:  # Marked with X
                var_name = f"v_{r}_{c}"
                var = Variable(var_name, domain={0, 1}, row=r, col=c)
                var.assign(0)  # Set as not filled
                variables[var_name] = var
    
    # Create constraint for each row
    for r in range(rows):
        # Get all variables in this row, including those already assigned
        row_vars = []
        for c in range(cols):
            var_name = f"v_{r}_{c}"
            if var_name in variables:
                row_vars.append(variables[var_name])
            else:
                # Create a new variable for consistency in the constraint
                var = Variable(var_name, domain={0, 1}, row=r, col=c)
                var.assign(0)  # Default to not filled
                row_vars.append(var)
                variables[var_name] = var
                
        # Create run length constraint for this row
        row_constraint = RunLengthsConstraint(
            set(row_vars),  # Convert to set for the constraint
            row_hints[r],   # Run lengths for this row
            variable_order=row_vars,  # Ordered list for sequence
            row=r, 
            col=None
        )
        constraints.append(row_constraint)
    
    # Create constraint for each column
    for c in range(cols):
        # Get all variables in this column, including those already assigned
        col_vars = []
        for r in range(rows):
            var_name = f"v_{r}_{c}"
            if var_name in variables:
                col_vars.append(variables[var_name])
            else:
                # Create a new variable for consistency in the constraint
                var = Variable(var_name, domain={0, 1}, row=r, col=c)
                var.assign(0)  # Default to not filled
                col_vars.append(var)
                variables[var_name] = var
                
        # Create run length constraint for this column
        col_constraint = RunLengthsConstraint(
            set(col_vars),  # Convert to set for the constraint
            col_hints[c],   # Run lengths for this column
            variable_order=col_vars,  # Ordered list for sequence
            row=None, 
            col=c
        )
        constraints.append(col_constraint)
    
    # # At least one of the first M positions must be filled
    # for r in range(rows):
    #     for run_index, run_length in enumerate(row_hints[r]):
    #         max_start = cols - run_length - (len(row_hints[r]) - run_index - 1)
    #         first_m_vars = [variables[f"v_{r}_{c}"] for c in range(max_start+1) 
    #                        if f"v_{r}_{c}" in variables]
    #         if first_m_vars:
    #             constraints.append(InequalityConstraint(set(first_m_vars), 0, greater_than=True))
    
    # for r in range(rows):
    #     row_vars = [variables[f"v_{r}_{c}"] for c in range(cols) if f"v_{r}_{c}" in variables]
    #     min_filled = sum(row_hints[r])
    #     constraints.append(InequalityConstraint(set(row_vars), min_filled-1, greater_than=True))
    
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
    """Class representing a Nonogram game state."""
    
    def __init__(self, board: np.ndarray, row_hints: List[List[int]], col_hints: List[List[int]]):
        """Initialize game from a board and run length hints.
        
        Args:
            board: rows x cols numpy array with:
                -1: unmarked cell
                0: X marker (definitely not filled)
                1: filled (black) cell
            row_hints: List of run length hints for each row
            col_hints: List of run length hints for each column
        """
        self.rows, self.cols = board.shape
        self.board = board.copy()
        self.initial_board = board.copy()
        self.row_hints = row_hints.copy() if row_hints is not None else []
        self.col_hints = col_hints.copy() if col_hints is not None else []
    
    def unmarked_squares_remaining(self):
        """Return the number of unmarked squares on the board."""
        return np.sum(self.board == -1)
        
    def is_solved(self) -> bool:
        """Check if the game is solved - all runs match their hints."""
        # Check if any unmarked cells remain
        if np.any(self.board == -1):
            return False
            
        # Check rows
        for r in range(self.rows):
            runs = []
            current_run = 0
            for c in range(self.cols):
                if self.board[r, c] == 1:  # Black cell
                    current_run += 1
                elif current_run > 0:
                    runs.append(current_run)
                    current_run = 0
            
            # Don't forget to add the last run if we ended with filled cells
            if current_run > 0:
                runs.append(current_run)
                
            # Check if runs match the hints
            if runs != self.row_hints[r]:
                return False
                
        # Check columns
        for c in range(self.cols):
            runs = []
            current_run = 0
            for r in range(self.rows):
                if self.board[r, c] == 1:  # Black cell
                    current_run += 1
                elif current_run > 0:
                    runs.append(current_run)
                    current_run = 0
            
            # Don't forget to add the last run if we ended with filled cells
            if current_run > 0:
                runs.append(current_run)
                
            # Check if runs match the hints
            if runs != self.col_hints[c]:
                return False
                
        return True
    
    def reset(self):
        """Reset the board to initial state."""
        self.board = self.initial_board.copy()
        
    def __str__(self) -> str:
        """Return string representation of the board."""
        result = []
        
        # Find the maximum number of hints in any row to format the hints section
        max_row_hints = max(len(hints) for hints in self.row_hints)
        
        # Add column hints at the top
        for hint_row in range(max(len(hints) for hints in self.col_hints)):
            row = " " * (max_row_hints * 3 + 1) + "|"
            for c in range(self.cols):
                hints = self.col_hints[c]
                if hint_row >= len(hints):
                    row += "  "  # Empty space if no hint
                else:
                    # Get the hint from bottom to top
                    index = len(hints) - 1 - hint_row
                    row += f"{hints[index]:2d}"
            result.append(row)
        
        # Add separator
        result.append("-" * (max_row_hints * 3 + 1) + "+" + "--" * self.cols)
        
        # Add board with row hints
        for r in range(self.rows):
            hints = self.row_hints[r]
            # Add row hints padding with spaces
            hint_str = " ".join(f"{h:2d}" for h in hints).rjust(max_row_hints * 3)
            row = f"{hint_str} |"
            
            for c in range(self.cols):
                cell = self.board[r, c]
                if cell == 1:  # Black cell
                    row += " ■"
                elif cell == 0:  # X marker
                    row += " ×"
                else:  # Unmarked
                    row += " ·"
            result.append(row)
            
        return "\n".join(result)
        
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unmarked cells."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == -1]
                
    def mark_filled(self, row: int, col: int):
        """Mark a cell as filled (black)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        self.board[row, col] = 1
        
    def mark_x(self, row: int, col: int):
        """Mark a cell with an X (not filled)."""
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


def generate_random_nonogram(rows: int, cols: int, filled_probability: float = 0.5):
    """Generate a random valid Nonogram puzzle.
    
    The algorithm creates a solution by randomly filling cells,
    then calculates the run length hints for rows and columns.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        filled_probability: Probability of a cell being filled
        
    Returns:
        board: Numpy array representing the solution
        row_hints: List of run length hints for each row
        col_hints: List of run length hints for each column
    """
    # Create a random board
    board = np.zeros((rows, cols), dtype=int)

    cells_to_fill = int(rows * cols * filled_probability)
    fileld_cells = random.sample(range(rows * cols), cells_to_fill)
    for cell in fileld_cells:
        board[cell // cols, cell % cols] = 1
    
    # Calculate row hints
    row_hints = []
    for r in range(rows):
        runs = []
        current_run = 0
        for c in range(cols):
            if board[r, c] == 1:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        # If no runs, add a 0
        if not runs:
            runs = [0]
        row_hints.append(runs)
    
    # Calculate column hints
    col_hints = []
    for c in range(cols):
        runs = []
        current_run = 0
        for r in range(rows):
            if board[r, c] == 1:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        # If no runs, add a 0
        if not runs:
            runs = [0]
        col_hints.append(runs)
    
    # Create an empty board for game play
    empty_board = np.full((rows, cols), -1)
    
    return empty_board, row_hints, col_hints, board  


def print_board(board, row_hints, col_hints):
    """Print a Nonogram board."""
    game = Game(board, row_hints, col_hints)
    print(game)
    print()


def generate_puzzle_from_solution(solution_board):

    rows, cols = solution_board.shape
    puzzle_board = np.full((rows, cols), -1, dtype=int)
    return puzzle_board


def check_unique_solution(puzzle_board, row_hints, col_hints, max_size=10000):
    """Check if a nonogram puzzle has a unique solution.
    
    Args:
        puzzle_board: The puzzle board with some cells revealed
        row_hints: Run length hints for each row
        col_hints: Run length hints for each column
        max_solutions: Maximum number of solutions to search for
        
    Returns:
        is_unique: True if the puzzle has exactly one solution
        solutions: List of found solutions (up to max_solutions)
    """
    constraints = board_to_constraints(puzzle_board, row_hints, col_hints)
    #constraints = sort_constraints_by_relatedness(constraints)
    constraints = sorted(constraints, key=lambda x: x.initial_size(), reverse=False)
    
    assignments = []
    for constraint in constraints:
        assignments = integrate_new_constraint(assignments, constraint)
    
        if (not assignments) or (not assignments[0]):
            return False, []
        size = len(assignments) * len(assignments[0])
        if size > max_size:
            return False, []
        
    return len(assignments) == 1, assignments


def generate_nonogram_puzzle(rows: int, cols: int, filled_probability: float = 0.5, max_attempts: int = 50):
    """Generate a Nonogram puzzle with a unique solution.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        filled_probability: Probability of a cell being filled
        max_attempts: Maximum number of attempts to generate a valid puzzle
        
    Returns:
        puzzle_board: The puzzle board with -1 for unmarked cells
        solution_board: The solution board with 0s and 1s
        row_hints: Run length hints for each row
        col_hints: Run length hints for each column
    """
    for _ in range(max_attempts):
        # Generate a random solution
        empty_board, row_hints, col_hints, solution_board = generate_random_nonogram(
            rows, cols, filled_probability)
        
        # Start with no cells revealed
        puzzle_board = np.full((rows, cols), -1, dtype=int)
        
        # Check if empty puzzle has a unique solution based just on the hints
        is_unique, solutions = check_unique_solution(puzzle_board, row_hints, col_hints)
        
        if is_unique:
            # We found a puzzle with a unique solution
            return puzzle_board, solution_board, row_hints, col_hints
    
    # If we failed after max_attempts, return None
    return None, None, None, None


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

    
def solve_board(board, row_sums, col_sums, max_size=2500):
    """Solve a Nonogram board.
    
    Returns the solution board.
    """
    constraints = board_to_constraints(board, row_sums, col_sums)
    constraints = sort_constraints_by_relatedness(constraints)

    small_constraints = sort_constraints_by_relatedness([c for c in constraints if c.initial_size() <= 10])
    large_constraints = sort_constraints_by_relatedness([c for c in constraints if c.initial_size() > 10])
    constraints = small_constraints + large_constraints
    #constraints = sorted(constraints, key=lambda x: x.initial_size(), reverse=False)
    
    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    for i, constraint in enumerate(constraints):
        new_assignments = integrate_new_constraint(assignments, constraint, max_size=max_size)
        if new_assignments is None or len(new_assignments) == 0:
            return np.array(board)
        elif len(new_assignments) > max_size:
            new_assignments = new_assignments[:max_size]

        assignments = new_assignments

        if i % 10 == 0:
            print(f"Run {i}/{len(constraints)} constraints, {len(assignments)} assignments")
            solved_board = np.array(board)
            solved_variables = get_solved_variables(assignments)
            for v in solved_variables:
                solved_board[v.row][v.col] = solved_variables[v]
            print_board(solved_board, row_sums, col_sums)
        if assignments and len(assignments) == 1:
            solved_variables = get_solved_variables(assignments)
            if len(solved_variables) == len(variables):
                break

    
    solved_board = np.array(board)
    solved_variables = get_solved_variables(assignments)

    for v in solved_variables:
        solved_board[v.row][v.col] = solved_variables[v]
    return solved_board
    

def run_nonogram_simulation(max_steps, unsolved_board, solved_board, row_hints, col_hints, 
                          memory_capacity, R_init, delta_R, ILtol_init, delta_IL, gamma):
    constraints = board_to_constraints(unsolved_board, row_hints, col_hints)
    # Map true assignments from solved_board to avoid re-solving
    variables = set().union(*[c.variables for c in constraints])
    mapped = {}
    for v in variables:
        if v.row is not None and v.col is not None:
            mapped[v] = solved_board[v.row][v.col]
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

def get_difficulty(unsolved_board, solved_board, row_hints, col_hints, 
                   n_simulations=10, memory_capacity=12, R_init=0.25, delta_R=0, ILtol_init=np.inf, delta_IL=0, gamma=1.0,
                   max_steps = 100):
    """Get the difficulty of a Nonogram puzzle.
    
    Args:
        unsolved_board: The initial puzzle board
        solved_board: The solution board
        row_hints: Run length hints for each row
        col_hints: Run length hints for each column
        n_simulations: Number of simulations to run
        memory_capacity: Agent's memory capacity
        search_budget: Agent's search budget    

    Returns:
        The difficulty of the puzzle.
    """
    error_rates = []
    for _ in range(n_simulations):
        error_rate = run_nonogram_simulation(max_steps, unsolved_board, solved_board, row_hints, col_hints, 
                          memory_capacity, R_init, delta_R, ILtol_init, delta_IL, gamma)
        error_rates.append(error_rate)
    return np.mean(error_rates)

def generate_boards_by_difficulty(min_difficulty, max_difficulty, n_boards, rows, cols, n_priority = 100):
    boards = []
    n_attempts = 0
    while len(boards) < n_boards:
        if n_attempts % 10 == 0:
            print(f"Current difficulty range: {min_difficulty:.2f} to {max_difficulty:.2f}")
        n_attempts += 1
        if n_attempts % 100 == 0:
            min_difficulty *= 0.95
            max_difficulty *= 1.05
            print(f"Adjusting difficulty range to {min_difficulty:.2f} to {max_difficulty:.2f}")

        filled_probability = np.random.uniform(0.2, 0.7)
        puzzle_board, solution_board, row_hints, col_hints = generate_nonogram_puzzle(rows, cols,
                                                              filled_probability=filled_probability)
        if puzzle_board is None or solution_board is None:
            print("Failed to generate a puzzle with a unique solution.")
            continue
        difficulty = get_difficulty(puzzle_board, solution_board, row_hints, col_hints,
                                    n_simulations=10, memory_capacity=10, R_init=0.25, delta_R=0, ILtol_init=np.inf, delta_IL=0, gamma=1.0,
                                    max_steps = 200)
        if difficulty >= min_difficulty and difficulty <= max_difficulty:
            boards.append({
                "rows": rows,
                "cols": cols,
                "id": hash_board_state(puzzle_board, row_hints, col_hints),
                "priority": 1*(len(boards) < n_priority),
                "game_state": puzzle_board.tolist(),
                "game_board": solution_board.tolist(),
                "n_black_cells": int(np.sum(solution_board == 1)),
                "row_hints": row_hints,
                "col_hints": col_hints,
                "difficulty": difficulty
            })
            print(f"Found board {len(boards)}/{n_boards} with difficulty {difficulty:.2f}")
            print_board(puzzle_board, row_hints, col_hints)
            print_board(solution_board, row_hints, col_hints)
        else:
            print(f"Skipping board {len(boards)}/{n_boards} with difficulty {difficulty:.2f}")

    boards = sorted(boards, key=lambda x: x["difficulty"])
    for i, board in enumerate(boards):
        board["idx"] = i
        print(f"Board {i+1} (difficulty: {board['difficulty']:.2f}):")
        print_board(np.array(board["game_state"]), board["row_hints"], board["col_hints"])
        print_board(np.array(board["game_board"]), board["row_hints"], board["col_hints"])

    return boards

def save_boards(boards: List[dict], filename: str, filepath=None):
    """Save multiple Light Up boards to a single JSON file."""
    if filepath is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'nonograms')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(boards, f, indent=2)

def load_boards(filename: str, filepath=None):
    """Load multiple Light Up boards from a JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'nonograms', f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    
    with open(filepath, 'r') as f:
        return json.load(f)
    



if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))


        
    rows, cols = 5,5
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
        min_difficulty = 0.35    
        max_difficulty = 1.0
    else:
        min_difficulty = 0.0
        max_difficulty = 1.0

    if for_website:
        filepath = os.path.join(current_dir, 'puzzle_website')
        if difficulty == "easy":
            filename = f"nonograms_{rows}x{cols}_easy"
        elif difficulty == "hard":
            filename = f"nonograms_{rows}x{cols}_hard"
        else:
            filename = f"nonograms_{rows}x{cols}_expert"
    else:
        filepath = os.path.join(current_dir, 'saved_boards', 'nonograms')
        filename = f"nonograms_{rows}x{cols}"

        

    print(f"Generating a {rows}x{cols} nonogram puzzle...")

    boards = generate_boards_by_difficulty(min_difficulty, max_difficulty, n_boards, rows, cols)
    save_boards(boards, filename, filepath)
