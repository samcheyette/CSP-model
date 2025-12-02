import os
import numpy as np
from typing import List, Tuple
from models.CSP_working_model.constraints import *
from models.CSP_working_model.special_constraints.sudoku_constraints import UniquenessConstraint
from models.CSP_working_model.utils.assignment_utils import\
     integrate_new_constraint, integrate_constraints, get_solved_variables
import math
import random
import json



def board_to_constraints(board: np.ndarray) -> List[Constraint]:
    """Convert board state to list of constraints.
    
    Creates variables for each empty cell and uniqueness constraints for:
    - Each row
    - Each column
    - Each box
    
    Empty cells are represented by either 0 or -1.
    """
    constraints = []
    size = board.shape[0]
    box_size = int(math.sqrt(size))
    
    # Create variables for each empty cell
    variables = {}
    for r in range(size):
        for c in range(size):
            if board[r, c] <= 0:  # Empty cell (either 0 or -1)
                variables[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", domain=set(range(1, size+1)), row=r, col=c)
    
    # Add row constraints
    for r in range(size):
        # Get variables and constants in this row
        row_vars = [variables[f"v_{r}_{c}"] for c in range(size) 
                   if f"v_{r}_{c}" in variables]
        constants = {board[r, c] for c in range(size) 
                    if board[r, c] > 0}  # Given numbers
        if row_vars:  # Only create constraint if there are variables
            constraints.append(UniquenessConstraint(row_vars, constants=constants, row=r, col=None))
    
    # Add column constraints
    for c in range(size):
        # Get variables and constants in this column
        col_vars = [variables[f"v_{r}_{c}"] for r in range(size) 
                   if f"v_{r}_{c}" in variables]
        constants = {board[r, c] for r in range(size) 
                    if board[r, c] > 0}  # Given numbers
        if col_vars:
            constraints.append(UniquenessConstraint(col_vars, constants=constants, row=None, col=c))
    
    # Add box constraints
    for box_r in range(0, size, box_size):
        for box_c in range(0, size, box_size):
            # Get variables and constants in this box
            box_vars = []
            constants = set()
            for r in range(box_r, box_r + box_size):
                for c in range(box_c, box_c + box_size):
                    if f"v_{r}_{c}" in variables:
                        box_vars.append(variables[f"v_{r}_{c}"])
                    elif board[r, c] > 0:
                        constants.add(board[r, c])
            if box_vars:
                constraints.append(UniquenessConstraint(box_vars, constants=constants))

    return constraints


def board_to_partial_constraints(board, max_constraint_size = 4, 
                                subset_size = 3, coverage_probability = 1.0):

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
    """Class representing a Sudoku game state."""
    
    def __init__(self, board: np.ndarray):
        """Initialize game from a board.
        
        Args:
            board: nxn numpy array with:
                -1 or 0: empty cell
                1-n: filled cell
            where n must be a perfect square (4, 9, 16, etc)
        """
        rows, cols = board.shape
        if rows != cols:
            raise ValueError(f"Board must be square, got {rows}x{cols}")
            
        # Check if size is perfect square
        box_size = int(math.sqrt(rows))
        if box_size * box_size != rows:
            raise ValueError(f"Board size must be perfect square, got {rows}")
            
        self.size = rows
        self.box_size = box_size
        
        # Create copies of the board, converting any 0s to -1s for consistency
        self.initial_board = board.copy()
        self.board = board.copy()
        
        # Convert 0s to -1s for unmarked cells
        self.initial_board[self.initial_board == 0] = -1
        self.board[self.board == 0] = -1


    def unmarked_squares_remaining(self):
        """Return the number of unmarked squares on the board."""
        return np.sum((self.board <= 0))
        
    def is_solved(self) -> bool:
        """Check if the game is solved."""
        valid_values = set(range(1, self.size + 1))
        
        # Convert board to absolute values for checking
        abs_board = np.abs(self.board)
        
        # Check all cells are filled (no 0s or -1s)
        if np.any(abs_board <= 0):
            return False
            
        # Check rows
        for row in abs_board:
            if set(row) != valid_values:
                return False
                
        # Check columns
        for col in abs_board.T:
            if set(col) != valid_values:
                return False
                
        # Check boxes
        for i in range(0, self.size, self.box_size):
            for j in range(0, self.size, self.box_size):
                box = abs_board[i:i+self.box_size, j:j+self.box_size].flatten()
                if set(box) != valid_values:
                    return False
                    
        return True
    
    def reset(self):
        """Reset the board to initial state.
        
        Ensures all unmarked cells use -1, not 0.
        """
        self.board = self.initial_board.copy()
        # Convert any remaining 0s to -1s for consistency
        self.board[self.board == 0] = -1
        
    def __str__(self) -> str:
        """Return string representation of the board."""
        cell_width = len(str(self.size))  # Width needed for largest number
        box_width = self.box_size * (cell_width + 1) + 1
        
        # Add horizontal lines between boxes
        rows = []
        for i, row in enumerate(self.board):
            if i > 0 and i % self.box_size == 0:
                rows.append('-' * (box_width * self.box_size - 1))
            
            # Add vertical lines between boxes
            row_str = ''
            for j, cell in enumerate(row):
                if j > 0 and j % self.box_size == 0:
                    row_str += '|'
                # Convert negative numbers to positive and handle display
                value = abs(cell) if cell != 0 and cell != -1 else 0
                display = str(value) if value != 0 else '.'
                if cell < 0 and cell != -1:  # Player-placed numbers shown in parentheses
                    display = f"({value})"
                row_str += ' ' * (cell_width - len(display) + 1)
                row_str += display
            rows.append(row_str)
            
        return '\n'.join(rows) + '\n'
        
    def __repr__(self) -> str:
        return f"Game(\n{str(self)}\n)"

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of all empty cells"""
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.board[r, c] <= 0]

    def get_player_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of all player-placed numbers"""
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.board[r, c] < 0 and self.board[r, c] != -1]

    def place_number(self, row: int, col: int, number: int):
        """Place a number at the specified position"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if not (1 <= number <= self.size):
            raise ValueError(f"Number {number} must be between 1 and {self.size}")
        if self.initial_board[row, col] > 0:  # Can't modify given numbers
            raise ValueError(f"Cannot modify given number at ({row}, {col})")
        self.board[row, col] = -number  # Store as negative to indicate player-placed

    def clear_cell(self, row: int, col: int):
        """Clear a player-placed number from a cell"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.initial_board[row, col] != 0 and self.initial_board[row, col] != -1:  # Can't modify given numbers
            raise ValueError(f"Cannot clear given number at ({row}, {col})")
        if self.board[row, col] == -1 or self.board[row, col] == 0:  # Can only clear player-placed numbers
            raise ValueError(f"No player-placed number to clear at ({row}, {col})")
        self.board[row, col] = -1  # Use -1 for unmarked cells

    def get_board_state(self):
        """Return current board state"""
        return self.board.copy()

    def get_value(self, row: int, col: int) -> int:
        """Get the value at the specified position
        
        Returns absolute value (positive) of the number in the cell.
        Returns 0 for unmarked cells (-1).
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        value = self.board[row, col]
        if value == -1:  # Unmarked cell
            return 0
        return abs(value) if value < 0 else value

    def is_given(self, row: int, col: int) -> bool:
        """Check if the number at the position was given in the initial board"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        return self.initial_board[row, col] > 0

def save_boards(boards: List[dict], filename: str, filepath=None):
    """Save multiple Sudoku boards to a single JSON file."""
    if filepath is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'sudoku')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(boards, f, indent=2)

def load_boards(filename: str, filepath=None):
    """Load multiple Sudoku boards from a JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'sudoku', f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, 'r') as f:
        return json.load(f)

def print_board(board):
    """Return string representation of the board.
    
    Handles -1 as unmarked squares.
    """
    if board is None:
        return
    size = board.shape[0]
    box_size = int(math.sqrt(size))
    cell_width = len(str(size))  # Width needed for largest number
    
    # Add horizontal lines between boxes
    rows = []
    for i, row in enumerate(board):
        if i > 0 and i % box_size == 0:
            rows.append('-' * (box_size * (cell_width + 1) * box_size + box_size - 1))
        
        # Add vertical lines between boxes
        row_str = ''
        for j, cell in enumerate(row):
            if j > 0 and j % box_size == 0:
                row_str += '|'
            # Convert negative numbers to positive and handle display
            value = abs(cell) if cell != 0 and cell != -1 else 0
            display = str(value) if value != 0 else '.'
            if cell < 0 and cell != -1:  # Player-placed numbers shown in parentheses
                display = f"({value})"
            row_str += ' ' * (cell_width - len(display) + 1)
            row_str += display
        rows.append(row_str)
        
    print('\n'.join(rows) + '\n')

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



def solve_board(board, max_size=None, print_every=None):
    constraints = board_to_constraints(board)
    constraints = sort_constraints_by_relatedness(constraints)

    solved_board = board.copy()
    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    for i, constraint in enumerate(constraints):
        assignments = integrate_new_constraint(assignments, constraint, max_size=max_size, return_none_if_too_large=True)

        if print_every is not None and i % print_every == 0:
            print(f"Step {i}/{len(constraints)}, {len(assignments)} assignments")
            print(constraint)
            solved_board = np.array(board)
            solved_variables = get_solved_variables(assignments)
            for v in solved_variables:
                solved_board[v.row][v.col] = solved_variables[v]
            print_board(solved_board)

        # Check if we already have a unique solution
        if assignments and (len(assignments) == 1)  and (len(assignments[0]) == len(variables)):
            solved_variables = get_solved_variables(assignments)
            if len(solved_variables) == len(variables):
                break


    solved_board = board.copy()
    solved_variables = get_solved_variables(assignments)

    for v in solved_variables:
        if v.row is not None and v.col is not None and solved_variables[v] is not None:
            solved_board[v.row][v.col] = solved_variables[v]
            
    return solved_board

def run_sudoku_simulation( max_steps,
    unsolved_board,
    solved_board,
    board_args,
    memory_capacity=10,
    R_init=1,
    delta_R=0,
    ILtol_init=np.inf,
    delta_IL=0,
    gamma=1.0,
    true_assignments=None,
    **kwargs,):
    """Run a simulation of an agent solving a Sudoku puzzle.
    
    Avoids re-solving by mapping variables to values from the solved board when possible.
    Handles both 0 and -1 as representations for unmarked squares.
    """
    constraints = board_to_constraints(unsolved_board)

    # If no true_assignments passed, derive from solved_board by aligning variable (row,col)
    if true_assignments is None:
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
    agent_board = unsolved_board.copy()
    for v in agent.variables:
        if v.is_assigned():
            agent_board[v.row][v.col] = v.value
            if v.value != solved_board[v.row][v.col]:
                errors += 1
        else:

            errors += 1 - 1/(2**v.get_initial_entropy())



    return errors / len(agent.variables) if len(agent.variables) > 0 else 0



def _bit_count(x):
    return x.bit_count()


def _iter_bits(mask):
    while mask:
        b = mask & -mask
        yield (b.bit_length() - 1)
        mask ^= b


def _count_sudoku_solutions(board, limit = 2):
    size = board.shape[0]
    box_size = int(math.sqrt(size))
    full_mask = (1 << size) - 1

    rows = [0] * size
    cols = [0] * size
    boxes = [0] * size
    empties = []

    for r in range(size):
        for c in range(size):
            v = int(board[r, c])
            if v > 0:
                bit = 1 << (v - 1)
                rows[r] |= bit
                cols[c] |= bit
                b = (r // box_size) * box_size + (c // box_size)
                boxes[b] |= bit
            else:
                empties.append((r, c))

    solution_count = 0

    def backtrack():
        nonlocal solution_count
        if solution_count >= limit:
            return

        best_idx = -1
        best_mask = 0
        best_count = size + 1

        for idx, (r, c) in enumerate(empties):
            if board[r, c] > 0:
                continue
            b = (r // box_size) * box_size + (c // box_size)
            used = rows[r] | cols[c] | boxes[b]
            cand = full_mask ^ used
            cnt = _bit_count(cand)
            if cnt == 0:
                return
            if cnt < best_count:
                best_count = cnt
                best_mask = cand
                best_idx = idx
                if cnt == 1:
                    break

        if best_idx == -1:
            solution_count += 1
            return

        r, c = empties[best_idx]
        b = (r // box_size) * box_size + (c // box_size)
        cand = best_mask

        while cand and solution_count < limit:
            bit_pos = (cand & -cand).bit_length() - 1
            cand ^= (1 << bit_pos)
            v = bit_pos + 1
            bit = 1 << bit_pos

            if (rows[r] & bit) or (cols[c] & bit) or (boxes[b] & bit):
                continue

            board[r, c] = v
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

            backtrack()

            boxes[b] ^= bit
            cols[c] ^= bit
            rows[r] ^= bit
            board[r, c] = -1

    backtrack()
    return solution_count



def _solve_sudoku_one(board, randomize = True):
    size = board.shape[0]
    box_size = int(math.sqrt(size))
    full_mask = (1 << size) - 1

    rows = [0] * size
    cols = [0] * size
    boxes = [0] * size
    empties = []

    for r in range(size):
        for c in range(size):
            v = int(board[r, c])
            if v > 0:
                bit = 1 << (v - 1)
                rows[r] |= bit
                cols[c] |= bit
                b = (r // box_size) * box_size + (c // box_size)
                boxes[b] |= bit
            else:
                board[r, c] = -1
                empties.append((r, c))

    def backtrack():
        best_idx = -1
        best_mask = 0
        best_count = size + 1

        for idx, (r, c) in enumerate(empties):
            if board[r, c] > 0:
                continue
            b = (r // box_size) * box_size + (c // box_size)
            used = rows[r] | cols[c] | boxes[b]
            cand = full_mask ^ used
            cnt = _bit_count(cand)
            if cnt == 0:
                return False
            if cnt < best_count:
                best_count = cnt
                best_mask = cand
                best_idx = idx
                if cnt == 1:
                    break

        if best_idx == -1:
            return True

        r, c = empties[best_idx]
        b = (r // box_size) * box_size + (c // box_size)

        candidates = [pos + 1 for pos in _iter_bits(best_mask)]
        if randomize:
            random.shuffle(candidates)

        for v in candidates:
            bit = 1 << (v - 1)
            if (rows[r] & bit) or (cols[c] & bit) or (boxes[b] & bit):
                continue

            board[r, c] = v
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

            if backtrack():
                return True

            boxes[b] ^= bit
            cols[c] ^= bit
            rows[r] ^= bit
            board[r, c] = -1

        return False

    solved = backtrack()
    return board if solved else None


def _generate_puzzle_by_removal(solution_board, target_clues, symmetrical = True, symmetry_prob = 0.5):
    size = solution_board.shape[0]
    puzzle = solution_board.copy()

    coords = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(coords)

    def current_clues():
        return int(np.sum(puzzle > 0))

    for r, c in coords:
        if puzzle[r, c] <= 0:
            continue
        r2, c2 = (size - 1 - r, size - 1 - c)
        prev1 = int(puzzle[r, c])
        prev2 = int(puzzle[r2, c2])

        use_pair = symmetrical and (random.random() < symmetry_prob)

        # Attempt removal
        puzzle[r, c] = -1
        pair_removed = False
        if use_pair and (r2 != r or c2 != c) and puzzle[r2, c2] > 0:
            puzzle[r2, c2] = -1
            pair_removed = True

        # Validate uniqueness and clue budget
        if _count_sudoku_solutions(puzzle.copy(), limit=2) != 1 or current_clues() < target_clues:
            # Revert
            puzzle[r, c] = prev1
            if pair_removed:
                puzzle[r2, c2] = prev2

        if current_clues() <= target_clues:
            break

    # Ensure uniqueness at the end
    if _count_sudoku_solutions(puzzle.copy(), limit=2) != 1:
        return None
    return puzzle

def generate_random_solved_board(size=9):
    """Generate a random fully solved Sudoku board using backtracking."""
    board = np.full((size, size), -1, dtype=int)
    # Seed a random first row for variety when possible
    first_row = list(range(1, size + 1))
    random.shuffle(first_row)
    board[0] = np.array(first_row, dtype=int)
    solved = _solve_sudoku_one(board, randomize=True)
    if solved is None:
        # Fallback to empty start if seeded row failed
        board[:] = -1
        solved = _solve_sudoku_one(board, randomize=True)
    return solved

def generate_sudoku_puzzle(size=9, reveal_percentage=0.3, max_attempts=50, symmetry_prob = 0.75):
    """Generate a Sudoku puzzle with a unique solution via clue removal."""
    target_clues = int(round(size * size * reveal_percentage))
    for _ in range(max_attempts):
        solution_board = generate_random_solved_board(size)
        if solution_board is None:
            continue
        puzzle_board = _generate_puzzle_by_removal(solution_board, target_clues, symmetrical=True, symmetry_prob=symmetry_prob)
        if puzzle_board is not None:
            return puzzle_board, solution_board
    return None, None

#def attempt_culling(assignments, constraints):

def check_unique_solution(puzzle_board):
    """Check if a Sudoku puzzle has a unique solution using fast backtracking."""
    if puzzle_board is None:
        return False
    return _count_sudoku_solutions(np.array(puzzle_board, dtype=int), limit=2) == 1

def get_difficulty(unsolved_board, solved_board, n_simulations=10, max_steps=200, memory_capacity=12, 
                   R_init=1, delta_R=0, ILtol_init=np.inf, delta_IL=0, gamma=1.0):
    """Calculate the difficulty of a Sudoku puzzle.
    
    Handles both 0 and -1 as representations for unmarked squares.
    """
    error_rates = []
    for n_sim in range(n_simulations):
        error_rate = run_sudoku_simulation(
            max_steps=max_steps, 
            unsolved_board=unsolved_board, 
            solved_board=solved_board, 
            board_args={}, 
            memory_capacity=memory_capacity, 
            R_init=R_init, 
            delta_R=delta_R, 
            ILtol_init=ILtol_init, 
            delta_IL=delta_IL, 
            gamma=gamma
        )

        error_rates.append(error_rate)

    return np.mean(error_rates)




def generate_boards_by_difficulty(min_difficulty, max_difficulty, n_boards, size, n_priority = 100, symmetry_prob = 0.75):
    """Generate Sudoku boards within a specified difficulty range.
    
    Args:
        min_difficulty: Minimum difficulty score (0-1)
        max_difficulty: Maximum difficulty score (0-1)
        n_boards: Number of boards to generate
        size: Size of the Sudoku board (9 for 9x9)
        
    Returns:
        List of dictionaries with board data
    """
    boards = []
    n_attempts = 0
    while len(boards) < n_boards:
        if n_attempts % 10 == 0:
            print(f"Current difficulty range: {min_difficulty:.2f} to {max_difficulty:.2f}")
        n_attempts += 1
        if n_attempts % 1000 == 0:
            min_difficulty *= 0.95
            max_difficulty *= 1.05
            print(f"Adjusting difficulty range to {min_difficulty:.2f} to {max_difficulty:.2f}")

        reveal_percentage = np.random.uniform(0.1, 0.2)

        puzzle_board, solution_board = generate_sudoku_puzzle(
            size, reveal_percentage=reveal_percentage, max_attempts=1, symmetry_prob=symmetry_prob)

        # print("puzzle board:")
        # print_board(puzzle_board)
        # print("solution board:")
        # print_board(solution_board)
        # print()
            
        if puzzle_board is None or solution_board is None:
            print("Failed to generate a puzzle with a unique solution.")
            continue
            
        difficulty = get_difficulty(
            puzzle_board, solution_board, max_steps=200,
            n_simulations=10, memory_capacity=12, R_init=1, delta_R=0, ILtol_init=np.inf, delta_IL=0, gamma=1.0
        )
        
        if difficulty >= min_difficulty and difficulty <= max_difficulty:
            boards.append({
                "game_state": puzzle_board.tolist(),
                "game_board": solution_board.tolist(),
                "id": hash_board_state(puzzle_board),
                "priority": 1*(len(boards) < n_priority),
                "size": size,
                "rows": size,
                "cols": size,
                "difficulty": difficulty
            })
            print(f"Found board {len(boards)}/{n_boards} with {difficulty:.2f} difficulty and {reveal_percentage*100:.2f}% revealed")
            print_board(puzzle_board)
            print_board(solution_board)
        else:
            print(f"Skipping board with difficulty {difficulty:.2f}")

    boards = sorted(boards, key=lambda x: x["difficulty"])
    for i, board in enumerate(boards):
        board["idx"] = i
        n_revealed = np.sum(np.array(board["game_state"]) != -1)
        print(f"Board {i+1} (difficulty: {board['difficulty']:.2f}, {n_revealed}/{size*size} revealed):")
        print_board(np.array(board["game_state"]))
        print()

    return boards





if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

 

    size=4
    n_boards = 400
    difficulty = "expert"
    for_website = True


    if difficulty == "easy":
        min_difficulty = 0.0
        max_difficulty = 0.33
    elif difficulty == "hard":
        min_difficulty = 0.33
        max_difficulty = 0.66
    elif difficulty == "expert":
        min_difficulty = 0.66
        max_difficulty = 1.0
    else:
        min_difficulty = 0.0
        max_difficulty = 1.0

    if for_website:
        filepath = os.path.join(current_dir, 'puzzle_website')
        if difficulty == "easy":
            filename = f"sudoku_{size}x{size}_easy"
        elif difficulty == "hard":
            filename = f"sudoku_{size}x{size}_hard"
        else:
            filename = f"sudoku_{size}x{size}_expert"
    else:
        filepath = os.path.join(current_dir, 'saved_boards', 'sudoku')
        filename = f"sudoku_{size}x{size}"

    print(f"Generating a {size}x{size} sudoku puzzle...")

    boards = generate_boards_by_difficulty(min_difficulty, max_difficulty, n_boards, size, n_priority = 100)
    save_boards(boards, filename, filepath=filepath)
