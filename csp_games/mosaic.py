import sys
import os
# Ensure parent directory (CSP_working_model) is on sys.path for local imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from models.CSP_working_model.grammar import *
from typing import List, Tuple, Set
from agent import Agent
from constraints import EqualityConstraint
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

_constraint_cache = {}
_true_assignments_cache = {}

"""
Mosaic rules:
- fill the grid with black (1) and white (0) cells.
- a number in a cell specifies how many black cells are in the 3x3 block
  centered at that cell (i.e., all neighbors including the cell itself).


- -1: unknown (unmarked) cell (variable that can be black=1 or white=0)
- -3: pre-marked black cell (fixed to 1)
- -4: pre-marked white cell (fixed to 0)
- 0..9: clue cells; the cell itself can still be black or white in the solution,
        and is included in the 3x3 count constraint.
"""


def board_to_constraints(board, **kwargs):

    keep_irrelevant = kwargs.get("keep_irrelevant", False)

    constraints = []
    rows, cols = board.shape

    variables = {}
    for r in range(rows):
        for c in range(cols):
            variables[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", row=r, col=c)

    for r in range(rows):
        for c in range(cols):
            cell_value = board[r, c]
            if cell_value >= 0:  # clue cell (0..9)
                # 3x3 neighborhood including (r,c)
                neighbors = get_adjacent_coords(rows, cols, r, c)
                neighbors.append((r, c))  # include center

                relevant_variables = set()
                for nr, nc in neighbors:
                    relevant_variables.add(variables[f"v_{nr}_{nc}"])

                if relevant_variables or keep_irrelevant:
                    constraint = EqualityConstraint(relevant_variables, int(cell_value), row=r, col=c)
                    if constraint not in constraints:
                        constraints.append(constraint)

    # Apply any pre-marked assignments (-3 black, -4 white)
    for v in variables.values():
        if board[v.row, v.col] == -3:
            v.assign(1)
        elif board[v.row, v.col] == -4:
            v.assign(0)

    return constraints

def print_board(board, *args):
    symbols = {-1: '?', -3: 'F', -4: 'X'}
    rows = []
    for row in board:
        row_str = [symbols.get(cell, str(cell)) for cell in row]
        rows.append(' '.join(row_str))
    print('\n'.join(rows) + "\n")

def hash_board_state(board):
    import hashlib
    if isinstance(board, list):
        board = np.array(board)
    flat_board = board.flatten()
    board_str = f"{board.shape[0]}x{board.shape[1]}:" + ",".join(map(str, flat_board))
    hash_obj = hashlib.sha256(board_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


class Game:
    """Represents a Mosaic game state"""
    def __init__(self, board: np.ndarray):
        self.initial_board = board.copy()
        self.board = board.copy()  # Make a copy to avoid modifying original
        self.rows, self.cols = board.shape

    def get_adjacent_coords(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get coordinates of all valid adjacent squares (8-neighborhood)."""
        adjacent = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_r, new_c = row + dr, col + dc
                if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                    adjacent.append((new_r, new_c))
        return adjacent

    def get_unrevealed_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unrevealed squares (including pre-marked black/white)."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] in [-1, -3, -4]]

    def get_unmarked_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unmarked squares (no black/white mark)."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == -1]

    def get_revealed_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all revealed squares (with numbers)."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] >= 0]

    def get_neighboring_variables(self, row: int, col: int) -> Set[Variable]:
        """Get variables representing cells in the 3x3 neighborhood centered at (row,col)."""
        neighbors = self.get_adjacent_coords(row, col)
        neighbors.append((row, col))  # include center for Mosaic
        return [Variable(f"v_{r}_{c}") for r, c in neighbors
                if self.board[r, c] in [-1, -3, -4] or self.board[r, c] >= 0
                ]

    def get_revealed_neighbors(self, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get coordinates and values of revealed squares in the 3x3 neighborhood (including center)."""
        neighbors = self.get_adjacent_coords(row, col)
        neighbors.append((row, col))
        return [(r, c, self.board[r, c]) for r, c in neighbors if self.board[r, c] >= 0]

    def mark_black(self, row: int, col: int):
        """Mark a cell as black at the specified position (analogous to flag in minesweeper)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-1, -3, -4]:
            raise ValueError(f"Cannot mark revealed cell at ({row}, {col})")
        self.board[row, col] = -3

    def mark_white(self, row: int, col: int):
        """Mark a cell as white at the specified position (analogous to safe mark)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-1, -3, -4]:
            raise ValueError(f"Cannot mark revealed cell at ({row}, {col})")
        self.board[row, col] = -4

    def unmark(self, row: int, col: int):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-3, -4]:
            raise ValueError(f"Cannot unmark revealed cell at ({row}, {col})")
        self.board[row, col] = -1

    def get_board_state(self):
        """Return current board state"""
        return self.board.copy()

    def is_solved(self):
        """Check if the game is solved (no unknown cells)."""
        return np.sum(self.board == -1) == 0

    def reset(self):
        self.board = self.initial_board.copy()

    def __str__(self) -> str:
        """Return string representation of the board (same symbols as minesweeper)."""
        symbols = {
            -1: '?',  # unknown
            -3: 'F',  # black mark
            -4: 'X'   # white mark
        }

        rows = []
        for row in self.board:
            row_str = [symbols.get(cell, str(cell)) for cell in row]
            rows.append(' '.join(row_str))

        return '\n'.join(rows)

    def __repr__(self) -> str:
        return f"Game(\n{str(self)}\n)"


def get_board_state(agent, game_board, *args):
    """Convert current agent state to board representation (black=-3, white=-4)."""
    board = np.array(game_board, dtype=int)
    for v in agent.variables:
        if v.is_assigned():
            board[v.row][v.col] = -3 if v.value == 1 else -4
    return board

def solved_to_puzzle(board):
    board = np.array(board)
    return (board >= 0) * board + (board < 0) * -1

def save_boards(data, filename: str, filepath = None):
    """Save multiple Mosaic boards to a single JSON file."""
    if filepath is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'mosaic')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_boards(filename: str, filepath = None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'mosaic', f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data

def solve_board(board, max_size=10000, print_every=100):
    constraints = board_to_constraints(board)
        # For Mosaic, constraints can be large; break them up similar to Minesweeper
        # constraints = break_up_constraints(board_to_constraints(board),
    #                                    max_constraint_size=max_constraint_size,
    #                                    subset_size=subset_size,
    #                                    coverage_probability=coverage_probability)
    constraints = sort_constraints_by_relatedness(constraints)

    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    for i, constraint in enumerate(constraints):
        new_assignments = integrate_new_constraint(assignments, constraint, max_size=max_size)
        if new_assignments is None or len(new_assignments) == 0:
            return np.array(board)

        assignments = new_assignments
        if assignments and len(assignments) == 1:
            solved_variables = get_solved_variables(assignments)
            if len(solved_variables) == len(variables):
                break

    solved_board = np.array(board)
    solved_variables = get_solved_variables(assignments)

    for v in solved_variables:
        solved_board[v.row][v.col] = -3 if solved_variables[v] == 1 else -4
    return solved_board


def load_stimuli(path):
    with open(path, 'r') as file:
        stimuli = json.load(file)
    if stimuli:
        if isinstance(stimuli[0], list):
            stimuli = [np.array(stimulus) for stimulus in stimuli]
        else:
            stimuli = np.array([np.array(stimulus["game_state"]) for stimulus in stimuli])
        return stimuli


def run_mosaic_simulation(
    max_steps,
    unsolved_board,
    solved_board,
    board_args,
    memory_capacity=10,
    R_init=0.25,
    delta_R=0,
    ILtol_init=np.inf,
    delta_IL=0,
    gamma=1.0,
    constraints=None,
    true_assignments=None,
    assignment_cap=50000,
    **kwargs,
):
    # Allow callers to pass precomputed constraints/solutions; otherwise cache by board hash
    board_hash = hash_board_state(unsolved_board)

    if constraints is None:
        constraints = _constraint_cache.get(board_hash)
        if constraints is None:
            constraints = board_to_constraints(unsolved_board, **board_args)
            _constraint_cache[board_hash] = constraints

    if true_assignments is None:
        true_assignments = _true_assignments_cache.get(board_hash)
        if true_assignments is None:
            # Integrate with a cap to avoid combinatorial blowup
            assignments_local = []
            for c in constraints:
                assignments_local = integrate_new_constraint(assignments_local, c, max_size=assignment_cap)
                if not assignments_local:
                    break
            true_assignments = get_solved_variables(assignments_local) if assignments_local else {}
            _true_assignments_cache[board_hash] = true_assignments

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

    errors = 0.0
    total_vars = len(agent.variables)
    if total_vars == 0:
        return 0.0

    # Expect solved_board to encode black/white truth: -3 for black (1), -4 for white (0)
    for v in agent.variables:
        if v.is_assigned():
            true_is_black = (solved_board[v.row][v.col] == -3)
            if (v.value == 1) != true_is_black:
                errors += 1
        else:
            errors += 0.5

    return errors / total_vars


def get_difficulty(
    unsolved_board,
    solved_board,
    n_simulations=10,
    memory_capacity=10,
    max_steps=100,
    R_init=0.25,
    delta_R=0,
    ILtol_init=np.inf,
    delta_IL=0,
    gamma=1.0,
    assignment_cap=50000,
    constraints=None,
    true_assignments=None,
):
    # Use provided constraints/solutions when available; otherwise compute and cache once per board
    if constraints is None:
        constraints = _constraint_cache.get(hash_board_state(unsolved_board))
        if constraints is None:
            constraints = board_to_constraints(unsolved_board, **{})
            _constraint_cache[hash_board_state(unsolved_board)] = constraints

    if true_assignments is None:
        true_assignments = _true_assignments_cache.get(hash_board_state(unsolved_board))
        if true_assignments is None:
            assignments_local = []
            for c in constraints:
                assignments_local = integrate_new_constraint(assignments_local, c, max_size=assignment_cap)
                if not assignments_local:
                    break
            true_assignments = get_solved_variables(assignments_local) if assignments_local else {}
            _true_assignments_cache[hash_board_state(unsolved_board)] = true_assignments

    error_rates = []
    for _ in range(n_simulations):
        error_rate = run_mosaic_simulation(
            max_steps=max_steps,
            unsolved_board=unsolved_board,
            solved_board=solved_board,
            board_args={},
            memory_capacity=memory_capacity,
            R_init=R_init,
            delta_R=delta_R,
            ILtol_init=ILtol_init,
            delta_IL=delta_IL,
            gamma=gamma,
            constraints=constraints,
            true_assignments=true_assignments,
            assignment_cap=assignment_cap,
        )
        error_rates.append(error_rate)

    return np.mean(error_rates)



def generate_random_mosaic_game(rows: int, cols: int, n_black: int,
                                require_unique: bool = True,
                                min_nontrivial: float = 0.9,
                                max_attempts: int = 50) -> Tuple[np.ndarray, np.ndarray]:

    def has_numbered_neighbor(puzzle, x, y):
        """Check if cell has at least one numbered neighbor (>=0) in its 3x3 neighborhood including itself."""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if puzzle[nx, ny] >= 0:
                        return True
        return False

    def is_valid_puzzle(puzzle):
        """Check if puzzle has no isolated unknown cells (each unknown touches a clue)."""
        for r in range(rows):
            for c in range(cols):
                if puzzle[r, c] == -1 and not has_numbered_neighbor(puzzle, r, c):
                    return False
        return True

    def create_complete_board():
        # Create black vector and shuffle it
        black_vector = np.array([1] * n_black + [0] * (rows * cols - n_black))
        np.random.shuffle(black_vector)
        black_matrix = black_vector.reshape((rows, cols))

        # Calculate numbers for all cells as 3x3 sum including center
        numbers = np.zeros_like(black_matrix, dtype=int)
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in shifts:
            shifted = np.roll(black_matrix, shift=(dx, dy), axis=(0, 1))

            # Clear wrapped edges (avoid toroidal wrap)
            if dx == -1: shifted[-1, :] = 0
            elif dx == 1: shifted[0, :] = 0
            if dy == -1: shifted[:, -1] = 0
            elif dy == 1: shifted[:, 0] = 0

            numbers += shifted

        return numbers, black_matrix



    def nontrivial_count(constraints):
        count = 0
        for c in constraints:
            if c.size() > 1:
                count += 1
        return count

    import random

    for _ in range(max_attempts):
        # Generate complete board with black/white and numbers
        numbers, black_matrix = create_complete_board()

        # Reveal some random cells using zero-expansion and random seeds
        revealed = np.zeros((rows, cols), dtype=bool)

        cell_locations = list(product(range(rows), range(cols)))

        reveal = random.sample(cell_locations, random.randint(len(cell_locations)//4, 3*len(cell_locations)//4))
        for x, y in reveal:
            revealed[x, y] = True


        # Ensure at least a few are revealed
        if not revealed.any():
            x, y = random.randrange(rows), random.randrange(cols)
            revealed[x, y] = True

        # Build puzzle with revealed numbers (clue layer)
        puzzle = np.full((rows, cols), -1, dtype=int)
        puzzle[revealed] = numbers[revealed]

        # Validate puzzle (no isolated unknowns)
        if not is_valid_puzzle(puzzle):
            continue

        # Build constraints and ensure nontriviality
        constraints = board_to_constraints(puzzle)
        if nontrivial_count(constraints) < min_nontrivial * max(1, len(constraints)):
            continue

        # Create full color board for ALL cells: black -> -3, white -> -4
        color_board = np.where(black_matrix == 1, -3, -4).astype(int)

        # Check for unique solution if required
        if require_unique:
            constraints = sort_constraints_by_relatedness(constraints)
            assignments = []
            for constraint in constraints:
                assignments = integrate_new_constraint(assignments, constraint, max_size=100000)
                if assignments is None or len(assignments) == 0:
                    break
            if assignments is not None and len(assignments) == 1:
                return puzzle, color_board
        else:
            return puzzle, color_board

    return None


def generate_boards_by_difficulty(min_difficulty, max_difficulty,
                min_black, max_black, n_boards, rows, cols, min_nontrivial = 0.0, n_priority = 100,
                initial_boards = None):
    boards = []

    if initial_boards is None:
        hashed_boards = set()
    else:
        hashed_boards = set(hash_board_state(board) for board in initial_boards)

    n_attempts = 0
    while len(boards) < n_boards:
        if n_attempts % 10 == 0:
            print(f"Current difficulty range: {min_difficulty:.2f} to {max_difficulty:.2f}")
        n_attempts += 1
        if n_attempts % 500 == 0:
            min_difficulty *= 0.95
            max_difficulty *= 1.05
            print(f"Adjusting difficulty range to {min_difficulty:.2f} to {max_difficulty:.2f}")

        n_black = np.random.randint(min_black, max_black+1)
        result = generate_random_mosaic_game(rows, cols, n_black, require_unique=True, min_nontrivial=min_nontrivial)

        if result is None:
            continue
        unsolved_board, solved_board = result  # puzzle (numbers) and color board
        hashed_board = hash_board_state(unsolved_board)
        if hashed_board in hashed_boards:
            print(f"Skipping board {len(boards)}/{n_boards} --- already seen!")
            print_board(unsolved_board)
            continue

        # Skip trivial (all unknowns same color); estimate via majority
        n_unknown = np.sum(unsolved_board == -1)
        if n_unknown > 0 and (n_black == n_unknown or (rows*cols - n_black) == n_unknown):
            print(f"Skipping board {len(boards)}/{n_boards} --- trivial board!")
            print_board(unsolved_board)
            print_board(solved_board)
            hashed_boards.add(hashed_board)
            continue


        # Precompute constraints and true assignments once and pass through to avoid repeated solving
        assignment_cap = 50000
        constraints = board_to_constraints(unsolved_board)
        assignments_local = []
        for c in constraints:
            assignments_local = integrate_new_constraint(assignments_local, c, max_size=assignment_cap)
            if not assignments_local:
                break
        true_assignments = get_solved_variables(assignments_local) if assignments_local else {}

        difficulty = get_difficulty(
            unsolved_board,
            solved_board,
            n_simulations=30,
            memory_capacity=10,
            assignment_cap=assignment_cap,
            constraints=constraints,
            true_assignments=true_assignments,
        )

        print(f"Generating board {len(boards)}/{n_boards} (difficulty: {min_difficulty:.2f} - {max_difficulty:.2f}) found: {difficulty:.2f}")

        if difficulty >= min_difficulty and difficulty <= max_difficulty:
            boards.append({
                "id": hash_board_state(unsolved_board),
                "priority": 1*(len(boards) < n_priority),
                "rows": rows,
                "cols": cols,
                "game_state": unsolved_board.tolist(),
                "game_board": solved_board.tolist(),
                "n_black": n_black,
                "difficulty": difficulty
            })
            print(f"Found board {len(boards)}/{n_boards} with difficulty {difficulty:.2f}")
            print_board(unsolved_board)
            print_board(solved_board)
            hashed_boards.add(hashed_board)
        else:
            print(f"Skipping board {len(boards)}/{n_boards} with difficulty {difficulty:.2f}")
            print_board(unsolved_board)
            print_board(solved_board)

    boards = sorted(boards, key=lambda x: x["difficulty"])

    idx_offset = len(initial_boards) if initial_boards is not None else 0
    for i, board in enumerate(boards):
        board["idx"] = i + idx_offset
        print(f"Board {i+1} (difficulty: {board['difficulty']:.2f}):")
        print_board(np.array(board["game_state"]))
        print_board(np.array(board["game_board"]))
        print()

    return boards


if __name__ == "__main__":
    rows = cols = size = 7
    n_boards = 400
    min_black = (size**2)//8
    max_black = (size**2)//3
    difficulty = "expert"


    current_dir = os.path.dirname(os.path.abspath(__file__))

    filepath = os.path.join(current_dir, 'puzzle_website')
    if difficulty == "easy":
        filename = f"mosaic_{rows}x{cols}_easy"
        min_difficulty = 0.00
        max_difficulty = 0.2
    elif difficulty == "hard":
        min_difficulty = 0.2
        max_difficulty = 1.0
    elif difficulty == "expert":
        min_difficulty = 0.3
        max_difficulty = 1.0
    else:
        min_difficulty = 0.0
        max_difficulty = 1.0



    print(f"Generating {n_boards} boards of size {rows}x{cols} with mines between {min_black} and {max_black}...")


    filename = f"mosaic_{rows}x{cols}_{difficulty}"
    boards = generate_boards_by_difficulty(min_difficulty, max_difficulty, min_black, max_black, n_boards, rows, cols)
    save_boards(boards, filename, filepath=filepath)
    print(f"\nSaved {len(boards)} boards to {filename}.json")
