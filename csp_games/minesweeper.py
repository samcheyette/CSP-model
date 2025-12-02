import sys
import os

from grammar import *
from agent import Agent
from constraints import *
import numpy as np
from utils.utils import *
from typing import List, Tuple, Set
import json
#from make_stimuli_mousetrack import make_stimulus
#from depth_predictor import DepthPredictor
from itertools import combinations

"""
NOTE: for the game boards, we use the following:
0-8:    number of mines nearby
-1:     unknown
-3:     flag placed
-4:     safe mark placed (we don't think there's a mine)

EXAMPLE:
[[-1 -1  2 -1 -1  1 -1]
[ 1  1  3 -1 -1  4  3]
[ 0  0  1 -1 -1 -1 -1]
[ 0  0  1  3 -1  4  2]
[ 0  0  0  1 -1  1  0]
[ 0  1  1  3 -1  2  0]
[ 0  1 -1 -1 -1  1  0]]
"""



def board_to_constraints(board, **kwargs):
    if "n_mines" in kwargs:
        n_mines = kwargs["n_mines"]
    else:
        n_mines = None
    if "keep_irrelevant" in kwargs:
        keep_irrelevant = kwargs["keep_irrelevant"]
    else:
        keep_irrelevant = False
    constraints = []
    rows, cols = board.shape

    variables = {}
    for r in range(rows):
        for c in range(cols):
            if board[r, c] < 0:
                variables[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", row=r, col=c)

    for r in range(rows):
        for c in range(cols):
            cell_value = board[r, c]
            if 0 <= cell_value <= 8:
                adjacent = get_adjacent_coords(rows, cols, r, c)
                relevant_variables = set()
                for adj_r, adj_c in adjacent:
                    if board[adj_r, adj_c] < 0:
                        relevant_variables.add(variables[f"v_{adj_r}_{adj_c}"])

                if relevant_variables or keep_irrelevant:
                    constraint = EqualityConstraint(relevant_variables, cell_value, row=r, col=c)
                    if constraint not in constraints:
                        constraints.append(constraint)

    if n_mines is not None:
        constraints.append(EqualityConstraint(set(variables.values()), n_mines))

    for v in variables.values():
        if board[v.row, v.col] == -3:
            v.assign(1)
        elif board[v.row, v.col] == -4:
            v.assign(0)

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
    """Represents a Minesweeper game state"""
    def __init__(self, board: np.ndarray):
        self.initial_board = board.copy()
        self.board = board.copy()  # Make a copy to avoid modifying original
        self.rows, self.cols = board.shape

    def get_adjacent_coords(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get coordinates of all valid adjacent squares"""
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
        """Get coordinates of all unrevealed squares (including flagged and marked safe)"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] in [-1, -3, -4]]

    def get_unmarked_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unmarked squares (no flag or safe mark)"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == -1]

    def get_revealed_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all revealed squares (with numbers)"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if 0 <= self.board[r, c] <= 8]

    def get_neighboring_variables(self, row: int, col: int) -> Set[Variable]:
        """Get variables representing unrevealed squares adjacent to given position"""
        neighbors = self.get_adjacent_coords(row, col)
        return {Variable(f"v_{r}_{c}") for r, c in neighbors
                if self.board[r, c] in [-1, -3, -4]}

    def get_revealed_neighbors(self, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get coordinates and values of revealed squares adjacent to given position.
        Returns list of (row, col, value) tuples."""
        neighbors = self.get_adjacent_coords(row, col)
        return [(r, c, self.board[r, c]) for r, c in neighbors
                if 0 <= self.board[r, c] <= 8]

    def flag(self, row: int, col: int):
        """Place a flag at the specified position"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-1, -3, -4]:  # Can only flag unknown cells
            raise ValueError(f"Cannot flag revealed cell at ({row}, {col})")
        self.board[row, col] = -3

    def mark_safe(self, row: int, col: int):
        """Mark a cell as safe (no mine) at the specified position"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-1, -3, -4]:  # Can only mark unknown cells
            raise ValueError(f"Cannot mark revealed cell at ({row}, {col})")
        self.board[row, col] = -4

    def unmark(self, row: int, col: int):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-3, -4]:  # Can only unmark flags and safe marks
            raise ValueError(f"Cannot unmark revealed cell at ({row}, {col})")
        self.board[row, col] = -1


    def get_board_state(self):
        """Return current board state"""
        return self.board.copy()

    def is_solved(self):
        """Check if the game is solved"""
        return np.sum(self.board == -1) == 0

    def reset(self):
        self.board = self.initial_board.copy()

    def __str__(self) -> str:
        """Return string representation of the board"""
        # Create a mapping for special values
        symbols = {
            -1: '?',  # unknown
            -3: 'F',  # flag
            -4: 'X'   # safe mark
        }

        # Build the string representation row by row
        rows = []
        for row in self.board:
            # Convert each cell to its string representation
            row_str = [symbols.get(cell, str(cell)) for cell in row]
            rows.append(' '.join(row_str))

        return '\n'.join(rows)

    def __repr__(self) -> str:
        return f"Game(\n{str(self)}\n)"



def load_stimuli(path):
    with open(path, 'r') as file:
        print(path)
        stimuli = json.load(file)
    # Convert each stimulus to numpy array and save the conversion
    if stimuli:
        if isinstance(stimuli[0], list):
            stimuli = [np.array(stimulus) for stimulus in stimuli]
        else:
            stimuli = np.array([np.array(stimulus["game_state"]) for stimulus in stimuli])
        return stimuli



def get_board_state(agent, game_board, *args):
    """Convert current agent state to board representation"""
    board = np.array(game_board, dtype=int)  # Convert to numpy array first

    # Then overlay agent's decisions
    for v in agent.variables:
        if v.is_assigned():
            if v.value == 1:  # Mine
                board[v.row][v.col] = -3  # Flag
            else:  # Safe
                board[v.row][v.col] = -4  # Safe mark

    return board



def randomly_complete_game(agent, game_state):
    """Randomly assign remaining unassigned variables"""
    unassigned = agent.get_unassigned_variables()
    for v in unassigned:
        # Random 0/1 assignment
        value = np.random.randint(2)
        v.assign(value)
        agent.solved_variables.add(v)
        # Update game state
        if value == 1:  # Mine
            game_state[v.row][v.col] = -3  # Flag
        else:  # Safe
            game_state[v.row][v.col] = -4  # Safe mark
    return game_state

def solve_board(board, max_size=10000,
                max_constraint_size=2, subset_size=2, coverage_probability=1, print_every=100):
    constraints = board_to_partial_constraints(board, max_constraint_size=max_constraint_size,
                                               subset_size=subset_size,
                                               coverage_probability=coverage_probability)
    constraints = sort_constraints_by_relatedness(constraints)

    variables = set().union(*[c.variables for c in constraints])
    assignments = []
    for i, constraint in enumerate(constraints):
        new_assignments = integrate_new_constraint(assignments, constraint, max_size=max_size)
        if new_assignments is None or len(new_assignments) == 0:
            return np.array(board)


        assignments = new_assignments

        # if i % print_every == 0:
        #     print(f"Run {i}/{len(constraints)} constraints, {len(assignments)} assignments")
        #     solved_variables = get_solved_variables(assignments)
        #     solved_board = np.array(board)
        #     for v in solved_variables:
        #         solved_board[v.row][v.col] = -3 if solved_variables[v] == 1 else -4
        #     print_board(solved_board)
        if assignments and len(assignments) == 1:
            solved_variables = get_solved_variables(assignments)
            if len(solved_variables) == len(variables):
                break

    solved_board = np.array(board)
    solved_variables = get_solved_variables(assignments)

    for v in solved_variables:
        solved_board[v.row][v.col] = -3 if solved_variables[v] == 1 else -4
    return solved_board

def load_games(path):

    with open(path, 'r') as file:
        games = json.load(file)
    # Convert each game board back to numpy array
    return [np.array(game) for game in games]

def generate_random_minesweeper_game(rows: int, cols: int, n_mines: int,
                                   require_unique: bool = True,
                                   min_nontrivial: float = 0.9,
                                   max_attempts: int = 1000,
                                   target_reveals_ratio: float = 0.35) -> np.ndarray:

    def has_numbered_neighbor(board, x, y):
        """Check if cell has at least one numbered neighbor (0-8)"""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if 0 <= board[nx, ny] <= 8:
                        return True
        return False

    def is_valid_puzzle(puzzle):
        """Check if puzzle has no isolated unknown cells"""
        for r in range(rows):
            for c in range(cols):
                if puzzle[r, c] == -1 and not has_numbered_neighbor(puzzle, r, c):
                    return False
        return True

    def create_complete_board():
        # Create mine vector and shuffle it
        mine_vector = np.array([1] * n_mines + [0] * (rows * cols - n_mines))
        np.random.shuffle(mine_vector)
        mine_matrix = mine_vector.reshape((rows, cols))

        # Calculate numbers for non-mine cells
        board = np.zeros(mine_matrix.shape, dtype=int)
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in shifts:
            shifted = np.roll(mine_matrix, shift=(dx, dy), axis=(0, 1))

            # Clear wrapped edges
            if dx == -1: shifted[-1, :] = 0
            elif dx == 1: shifted[0, :] = 0
            if dy == -1: shifted[:, -1] = 0
            elif dy == 1: shifted[:, 0] = 0

            board += shifted

        board[mine_matrix == 1] = -1
        return board, mine_matrix

    def reveal_cell(board, revealed, x, y):
        """Recursively reveal cells starting from (x,y)"""
        if revealed[x, y] or board[x, y] == -1:
            return

        revealed[x, y] = True

        # If cell is empty, reveal adjacent cells
        if board[x, y] == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        reveal_cell(board, revealed, nx, ny)

    def nontrivial_count(constraints):
        count = 0
        for c in constraints:
            if c.size() > 1:
                count += 1
        return count

    def _count_minesweeper_solutions(puzzle, limit=2):
        # Variables are unknown cells (-1). Flags (-3) treated as known mines; safe (-4) as known safe
        rN, cN = puzzle.shape
        unknown_coords = []
        index_of = {}
        for r in range(rN):
            for c in range(cN):
                if puzzle[r, c] == -1:
                    index_of[(r, c)] = len(unknown_coords)
                    unknown_coords.append((r, c))

        n_vars = len(unknown_coords)
        if n_vars == 0:
            # Fully revealed; check consistency trivially satisfied counts as 1 solution
            return 1

        # Build constraints from revealed numbers
        constraints = []  # each: {"unassigned": set(var_idx), "remaining": int}
        var_to_constraints = [[] for _ in range(n_vars)]

        for r in range(rN):
            for c in range(cN):
                val = puzzle[r, c]
                if 0 <= val <= 8:
                    neighbors = get_adjacent_coords(rN, cN, r, c)
                    known_mines = 0
                    vars_here = []
                    for nr, nc in neighbors:
                        cell = puzzle[nr, nc]
                        if cell == -1:
                            vars_here.append(index_of[(nr, nc)])
                        elif cell == -3:
                            known_mines += 1
                        # -4 known safe and numbers do not change target

                    remaining = val - known_mines
                    if remaining < 0:
                        return 0
                    if remaining > len(vars_here):
                        return 0
                    if vars_here:
                        cons = {"unassigned": set(vars_here), "remaining": remaining}
                        cid = len(constraints)
                        constraints.append(cons)
                        for vid in vars_here:
                            var_to_constraints[vid].append(cid)

        # Assignments: -1 unknown, 0 safe, 1 mine
        assign = [-1] * n_vars

        def undo(trail):
            # Revert in reverse order
            for item in reversed(trail):
                if item[0] == "var":
                    _, vid, prev = item
                    assign[vid] = prev
                else:
                    # constraint update
                    _, cid, vid, dec = item
                    constraints[cid]["unassigned"].add(vid)
                    if dec:
                        constraints[cid]["remaining"] += 1

        def apply_assign(vid, value, trail):
            prev = assign[vid]
            if prev != -1:
                return prev == value
            assign[vid] = value
            trail.append(("var", vid, prev))
            for cid in var_to_constraints[vid]:
                cons = constraints[cid]
                if vid in cons["unassigned"]:
                    cons["unassigned"].remove(vid)
                    dec = (value == 1)
                    if dec:
                        cons["remaining"] -= 1
                    trail.append(("cons", cid, vid, dec))
                    if cons["remaining"] < 0:
                        return False
                    if cons["remaining"] > len(cons["unassigned"]):
                        return False
            return True

        def propagate(trail):
            changed = True
            while changed:
                changed = False
                for cid, cons in enumerate(constraints):
                    remaining = cons["remaining"]
                    unassigned = cons["unassigned"]
                    if remaining < 0 or remaining > len(unassigned):
                        return False
                    if remaining == 0 and unassigned:
                        # all unassigned here must be 0
                        for vid in list(unassigned):
                            if assign[vid] == 1:
                                return False
                            if assign[vid] == -1:
                                if not apply_assign(vid, 0, trail):
                                    return False
                                changed = True
                    elif remaining == len(unassigned) and unassigned:
                        # all unassigned here must be 1
                        for vid in list(unassigned):
                            if assign[vid] == 0:
                                return False
                            if assign[vid] == -1:
                                if not apply_assign(vid, 1, trail):
                                    return False
                                changed = True
            return True

        def choose_var():
            # Pick an unassigned var with highest degree
            best = -1
            best_deg = -1
            for vid in range(n_vars):
                if assign[vid] == -1:
                    deg = len(var_to_constraints[vid])
                    if deg > best_deg:
                        best_deg = deg
                        best = vid
            return best

        def all_assigned():
            for v in assign:
                if v == -1:
                    return False
            return True

        def dfs(count):
            if count >= limit:
                return count
            trail = []
            if not propagate(trail):
                undo(trail)
                return count
            if all_assigned():
                count += 1
                undo(trail)
                return count
            vid = choose_var()
            # Try 0 then 1
            for val in (0, 1):
                trail2 = []
                if apply_assign(vid, val, trail2):
                    count = dfs(count)
                undo(trail2)
                if count >= limit:
                    break
            undo(trail)
            return count

        return dfs(0)

    for _ in range(max_attempts):
        # Generate complete board with mines and numbers
        complete_board, mine_matrix = create_complete_board()

        # Start from fully revealed numbers (mines remain -1), then hide numbers while preserving uniqueness
        puzzle = complete_board.copy()
        safe_coords = list(zip(*np.where(complete_board >= 0)))
        random.shuffle(safe_coords)

        target_reveals = int(round(target_reveals_ratio * rows * cols))

        def current_reveals():
            return int(np.sum(puzzle >= 0))

        # Greedy removal with uniqueness check
        for r, c in safe_coords:
            if current_reveals() <= target_reveals:
                break
            if puzzle[r, c] < 0:
                continue
            prev = int(puzzle[r, c])
            puzzle[r, c] = -1

            # Keep if puzzle stays unique and remains structurally valid
            if _count_minesweeper_solutions(puzzle, limit=2) != 1 or not is_valid_puzzle(puzzle):
                puzzle[r, c] = prev

        constraints = board_to_constraints(puzzle)

        if nontrivial_count(constraints) < min_nontrivial * len(constraints):
            continue


        safe_cells = (puzzle < 0) * (1-mine_matrix)
        mine_cells = (puzzle < 0) * mine_matrix
        solution = safe_cells * -4 + mine_cells * -3 + puzzle * (puzzle >= 0)
        # Check for unique solution if required
        if require_unique:
            n_solutions = _count_minesweeper_solutions(puzzle, limit=2)
            if n_solutions == 1:
                return solution
            else:
                print(f"Skipping board --- ({n_solutions} solutions found)!")
                print_board(puzzle)
                print_board(solution)
                continue
        else:
            return solution
    return None

def save_boards(data: List[np.ndarray], filename: str, filepath = None):
    """Save multiple minesweeper boards to a single JSON file."""
    if filepath is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'minesweeper')
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_boards(filename: str, filepath = None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'minesweeper', f"{filename}.json")
    else:
        filepath = os.path.join(filepath, f"{filename}.json")

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data




def solved_to_puzzle(board):
    board = np.array(board)
    return (board >= 0) * board + (board < 0) * -1

def run_minesweeper_simulation(
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
    **kwargs,
):
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


def make_FC_pairs(easy_boards, hard_boards):
    if (not easy_boards) or (not hard_boards):
        return [], [], []
    elif len(easy_boards) < len(hard_boards):
        hard_boards = hard_boards[:len(easy_boards)]
    elif len(easy_boards) > len(hard_boards):
        easy_boards = easy_boards[:len(hard_boards)]


    random.shuffle(easy_boards)
    random.shuffle(hard_boards)

    easy_easy_pairs = []
    easy_hard_pairs = []
    hard_hard_pairs = []

    for i in range(0,2*len(easy_boards)//3, 2):
        easy_easy_pairs.append((easy_boards[i], easy_boards[i+1]))
    for i in range(0,2*len(hard_boards)//3, 2):
        hard_hard_pairs.append((hard_boards[i], hard_boards[i+1]))
    for i in range(2*len(easy_boards)//3,len(easy_boards)):
        easy_hard_pairs.append((easy_boards[i], hard_boards[i]))

    return easy_easy_pairs, easy_hard_pairs, hard_hard_pairs





def get_difficulty(
    unsolved_board,
    solved_board,
    n_simulations=10,
    memory_capacity=10,
    max_steps=100,
    R_init=1.0,
    delta_R=0,
    ILtol_init=np.inf,
    delta_IL=0,
    gamma=1.0,
):
    error_rates = []
    for _ in range(n_simulations):
        error_rate = run_minesweeper_simulation(
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
        )
        error_rates.append(error_rate)

    return np.mean(error_rates)


def get_p_perfect(
    unsolved_board,
    solved_board,
    n_simulations=10,
    memory_capacity=12,
    max_steps=200,
    R_init=0.25,
    delta_R=0,
    ILtol_init=np.inf,
    delta_IL=0,
    gamma=0.5,
):
    perfect_rates = []
    for _ in range(n_simulations):
        error_rate = run_minesweeper_simulation(
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
        )
        perfect_rates.append(1*(error_rate == 0))

    return np.mean(perfect_rates)



def generate_boards_by_difficulty(min_difficulty, max_difficulty,
                min_mines, max_mines,n_boards, rows, cols, min_nontrivial = 0.0, n_priority = 100,
                 use_any_error = False,
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
        if n_attempts % 100 == 0:
            min_difficulty *= 0.95
            max_difficulty *= 1.05
            print(f"Adjusting difficulty range to {min_difficulty:.2f} to {max_difficulty:.2f}")
        n_mines = np.random.randint(min_mines, max_mines)
        solved_board = generate_random_minesweeper_game(rows, cols, n_mines, require_unique=True, min_nontrivial=min_nontrivial)





        if solved_board is None:
            print(f"Skipping board {len(boards)}/{n_boards} --- no solution found!")
            continue

        unsolved_board = solved_to_puzzle(solved_board)
        hashed_board = hash_board_state(unsolved_board)
        if hashed_board in hashed_boards:
            print(f"Skipping board {len(boards)}/{n_boards} --- already seen!")
            print_board(unsolved_board)
            continue

        if n_mines == len(np.where(unsolved_board == -1)[0]):
            print(f"Skipping board {len(boards)}/{n_boards} --- trivial board!")
            print_board(unsolved_board)
            print_board(solved_board)
            hashed_boards.add(hashed_board)


        difficulty = get_difficulty(unsolved_board, solved_board, n_simulations=15, memory_capacity=10)

        print(f"Generating board {len(boards)}/{n_boards} (difficulty: {min_difficulty:.2f} - {max_difficulty:.2f}) found: {difficulty:.2f}")

        if difficulty >= min_difficulty and difficulty <= max_difficulty:
            boards.append({
                "id": hash_board_state(unsolved_board),
                "priority": 1*(len(boards) < n_priority),

                "rows": rows,
                "cols": cols,
                "game_state": unsolved_board.tolist(),
                "game_board": solved_board.tolist(),
                "n_mines": n_mines,
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


    # constraints = board_to_partial_constraints(puzzle_board, max_constraint_size=2, subset_size=2, coverage_probability=1)
    # for c in constraints:
    #     print(c)
    #     print()

    # solved_board = solve_board(puzzle_board, max_size=500,
    #                            max_constraint_size=2, subset_size=2,
    #                            coverage_probability=1, print_every=1)
    # print_board(solved_board)


    current_dir = os.path.dirname(os.path.abspath(__file__))

    rows = cols = size = 9
    n_boards = 400
    min_mines = 16
    max_mines = 24




    # n_mines = 3
    difficulty =  "expert"
    for_website = True
    filename = f"minesweeper_{rows}x{cols}_{difficulty}"


    if difficulty == "easy":
        min_difficulty = 0.0
        max_difficulty = 0.1
    elif difficulty == "hard":
        min_difficulty = 0.2
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
            filename = f"minesweeper_{rows}x{cols}_easy"
        else:
            filename = f"minesweeper_{rows}x{cols}_hard"
    else:
        filepath = os.path.join(current_dir, 'saved_boards', 'minesweeper')
        filename = f"minesweeper_{rows}x{cols}"


    print(f"Generating {n_boards} boards of size {rows}x{cols} with mines between {min_mines} and {max_mines}...")


    filename = f"minesweeper_{rows}x{cols}_{difficulty}"
    boards = generate_boards_by_difficulty(min_difficulty, max_difficulty, min_mines, max_mines, n_boards, rows, cols,  min_nontrivial = 1.0)
    save_boards(boards, filename, filepath=filepath)
    print(f"\nSaved {len(boards)} boards to {filename}.json")
