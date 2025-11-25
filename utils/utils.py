from re import L
import sys
import os
from itertools import chain, combinations

# Get absolute path to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add paths
sys.path.extend([project_root])
sys.path.append(os.path.join(project_root, "stimuli"))
# from make_stimuli_mousetrack import make_stimulus
import json
import numpy as np
from grammar import *

# from constraints import *
from collections import defaultdict
from typing import List, Tuple
from scipy.special import gammaln
import random
from scipy.interpolate import interp1d
import pandas as pd


def get_powerset(iterable, min_size=1, max_size=None):
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return chain.from_iterable(
        combinations(s, r) for r in range(min_size, max_size + 1)
    )


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_variable_heatmap(constraints_in_path, game_board, subset=None):
    n_rows, n_cols = len(game_board), len(game_board[0])
    heatmap = np.zeros((n_rows, n_cols), dtype=int)

    for constraint in constraints_in_path:
        for v in constraint.get_unassigned():
            if subset is not None and v not in subset:
                continue
            heatmap[v.row][v.col] += 1
    return heatmap.tolist()


def get_constraint_heatmap(constraints_in_path, game_board):
    n_rows, n_cols = len(game_board), len(game_board[0])
    heatmap = np.zeros((n_rows, n_cols), dtype=int)

    for constraint in constraints_in_path:
        if constraint.row is not None and constraint.col is not None:
            heatmap[constraint.row][constraint.col] = 1
    return heatmap.tolist()


def get_row_col_clue_heatmap(constraints_in_path, game_board):
    n_rows, n_cols = len(game_board), len(game_board[0])
    row_heatmap = np.zeros(n_rows)
    col_heatmap = np.zeros(n_cols)

    for constraint in constraints_in_path:
        if constraint.row is not None and constraint.col is None:
            row_heatmap[constraint.row] = 1
        if constraint.col is not None and constraint.row is None:
            col_heatmap[constraint.col] = 1
    return row_heatmap.tolist(), col_heatmap.tolist()

def get_error_heatmap(initial_game_state, game_state, solution_board):
    """Create heatmap of errors (1 where wrong, 0 where correct or unsolved)"""
    heatmap = np.zeros_like(solution_board, dtype=int)

    for row in range(len(game_state)):
        for col in range(len(game_state[row])):
            if (
                initial_game_state[row][col] == -1
                and game_state[row][col] != -1
                and game_state[row][col] != solution_board[row][col]
            ):
                heatmap[row][col] = 1
    return heatmap.tolist()


def get_error_heatmap2(initial_game_state, game_state, solution_board):
    return (
        (
            (initial_game_state == -1)
            & (game_state != -1)
            & (game_state != solution_board)
        )
        .astype(int)
        .tolist()
    )


def get_correct_heatmap(initial_game_state, game_state, solution_board):
    """Create heatmap of correct marks (1 where correct, 0 where incorrect or unsolved)"""
    heatmap = np.zeros_like(solution_board, dtype=int)
    for row in range(len(game_state)):
        for col in range(len(game_state[row])):
            if (
                initial_game_state[row][col] == -1
                and game_state[row][col] != -1
                and game_state[row][col] == solution_board[row][col]
            ):
                heatmap[row][col] = 1
    return heatmap.tolist()


def get_correct_heatmap2(initial_game_state, game_state, solution_board):
    return (
        (
            (initial_game_state == -1)
            & (game_state != -1)
            & (game_state == solution_board)
        )
        .astype(int)
        .tolist()
    )


def get_marked_heatmap(initial_game_state, game_state):
    heatmap = np.zeros_like(game_state, dtype=int)
    for row in range(len(game_state)):
        for col in range(len(game_state[row])):
            if initial_game_state[row][col] == -1 and game_state[row][col] != -1:
                heatmap[row][col] = 1
    return heatmap.tolist()


def get_marked_heatmap2(initial_game_state, game_state):
    return ((initial_game_state == -1) & (game_state != -1)).astype(int).tolist()


def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_native_types(x) for x in obj]
    elif isinstance(obj, list):
        return [convert_to_native_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    return obj


def get_adjacent_coords(rows, cols, r, c):
    adjacent = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                adjacent.append((new_r, new_c))
    return adjacent


def get_revealed_squares(board):
    revealed = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            cell = board[row][col]
            if cell >= 0:
                revealed.append((row, col))

    return revealed


def get_unrevealed_squares(board):
    unrevealed = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            cell = board[row][col]
            if cell < 0:
                unrevealed.append((row, col))

    return unrevealed


def get_unmarked_squares(board):
    unmarked = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            cell = board[row][col]
            if cell == -1:
                unmarked.append((row, col))
    return unmarked


def find_components(items, get_neighbors):
    components = []
    unvisited = set(items)

    while unvisited:
        # Start a new component
        current = unvisited.pop()
        component = {current}
        queue = [current]

        # BFS to find all connected items
        while queue:
            item = queue.pop(0)
            for neighbor in get_neighbors(item):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def normalize(arr):
    return np.array(arr) / np.sum(arr)


def softmax(values, tau=1.0):
    arr = np.array(values) - np.max(values)
    if tau == 0:
        arr = 1.0 * (arr >= 0)
    elif tau == np.inf:
        arr = np.ones_like(arr)
    else:
        arr = np.exp(arr / tau)
    arr = normalize(arr)
    return arr


def softmax2(values, tau=1, sm=1e-10):
    arr = normalize(np.array(values) + sm)
    arr = arr ** (1 / tau)
    return normalize(arr)


def extend_model_traces(traces):
    max_steps = max([len(trace) for trace in traces])
    for trace in traces:
        if len(trace) < max_steps:
            trace.extend([trace[-1]] * (max_steps - len(trace)))
    return traces


def aggregate_traces(traces, metric="mean"):
    """Average multiple model traces into a single trace with mean predictions at each timestep."""
    traces = extend_model_traces(traces)
    avg_trace = []
    for step in range(len(traces[0])):
        if metric == "sum":
            mean_marked = np.sum(
                [trace[step]["marked_heatmap"] for trace in traces], axis=0
            )
            mean_correct = np.sum(
                [trace[step]["correct_heatmap"] for trace in traces], axis=0
            )
            mean_constraint_rt = np.sum(
                [trace[step]["constraint_rt_heatmap"] for trace in traces], axis=0
            )
            mean_variable_rt = np.sum(
                [trace[step]["variable_rt_heatmap"] for trace in traces], axis=0
            )
        else:
            mean_marked = np.mean(
                [trace[step]["marked_heatmap"] for trace in traces], axis=0
            )
            mean_correct = np.mean(
                [trace[step]["correct_heatmap"] for trace in traces], axis=0
            )
            mean_constraint_rt = np.mean(
                [trace[step]["constraint_rt_heatmap"] for trace in traces], axis=0
            )
            mean_variable_rt = np.mean(
                [trace[step]["variable_rt_heatmap"] for trace in traces], axis=0
            )

        avg_trace.append(
            {
                "step": step,
                "rt": np.mean([trace[step]["rt"] for trace in traces]),
                "marked_heatmap": mean_marked,
                "correct_heatmap": mean_correct,
                "constraint_rt_heatmap": mean_constraint_rt,
                "variable_rt_heatmap": mean_variable_rt,
                "cplx": np.mean([trace[step].get("cplx", 0) for trace in traces]),
                "IL": np.mean([trace[step].get("IL", 0) for trace in traces]),
                "IG": np.mean([trace[step].get("IG", 0) for trace in traces]),
            }
        )
    return avg_trace


def expand_df(df):
    """
    Expands a dataframe so each uniqueid has the same number of steps by copying the final state
    of games that finished early.
    """
    # Find the maximum number of steps across all games
    max_steps = df["step"].max()

    # Create a list to store expanded rows
    expanded_rows = []

    # Process each unique game
    for uniqueid in df["uniqueid"].unique():
        game_df = df[df["uniqueid"] == uniqueid].copy()
        current_steps = len(game_df)

        # Add all original rows
        expanded_rows.extend(game_df.to_dict("records"))

        if current_steps < max_steps + 1:  # +1 because steps start at 0
            # Get the final state of this game as a dictionary
            final_state = game_df.iloc[-1].to_dict()

            # Create additional rows by copying the final state
            for step in range(current_steps, max_steps + 1):
                new_row = final_state.copy()
                new_row["step"] = step
                expanded_rows.append(new_row)

    # Create new dataframe from expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    # Sort and reset index
    expanded_df = expanded_df.sort_values(["uniqueid", "step"]).reset_index(drop=True)

    return expanded_df


def aggregate_df(df, group_cols, scalar_cols, board_cols):
    """
    Aggregates a dataframe by computing means for scalar columns and board-shaped columns.
    """
    # First handle scalar columns using standard groupby
    scalar_agg = df.groupby(group_cols)[scalar_cols].mean()

    # Handle board columns
    board_aggs = []
    for group_values, group_df in df.groupby(group_cols):
        # Convert group_values to dict if it's a tuple
        if isinstance(group_values, tuple):
            group_dict = dict(zip(group_cols, group_values))
        else:
            group_dict = {group_cols[0]: group_values}

        # For each board column, compute cell-wise mean
        for board_col in board_cols:
            # Convert string representations to numpy arrays
            boards = []
            for board_str in group_df[board_col].values:
                if isinstance(board_str, str):
                    # Parse string representation of list
                    board = np.array(eval(board_str))
                else:
                    board = np.array(board_str)
                boards.append(board)

            # Stack boards and compute mean
            boards = np.stack(boards)
            mean_board = np.mean(boards, axis=0)

            # Add to group dict
            group_dict[board_col] = mean_board.tolist()

        board_aggs.append(group_dict)

    # Convert board aggregations to DataFrame
    board_df = pd.DataFrame(board_aggs)

    # Reset index on scalar aggregations
    scalar_agg = scalar_agg.reset_index()

    # Merge scalar and board results
    result = pd.merge(scalar_agg, board_df, on=group_cols)

    return result


def get_interpolated_prediction(states: np.array, times: np.array, ts):
    """Get interpolated prediction at time T."""
    # states (T, H, W)
    # times (T,)
    original_shape = states[0].shape

    flat_states = states.reshape(len(states), -1)  # (T, H*W)

    # Interpolate each cell
    interpolated_state = np.array(
        [np.interp(ts, times, flat_states[:, i]) for i in range(flat_states.shape[1])]
    ).T  # (Tnew, H*W)

    # Reshape back to original dimensions
    return interpolated_state.reshape((len(ts), *original_shape))  # (Tnew, H, W)


def get_averaged_interpolated_prediction(sequences, times_list, ts):
    """Get averaged interpolated prediction across multiple sequences."""
    # Validate inputs
    assert len(sequences) == len(times_list)
    n_sequences = len(sequences)

    # Get interpolated prediction for each sequence
    predictions = sum(
        (
            get_interpolated_prediction(np.array(states), np.array(times), ts)
            for states, times in zip(sequences, times_list)
        )
    )

    return predictions / n_sequences


def get_final_state_likelihood(board, model_traces, human_trace, epsilon):
    """Compute likelihood of final state for a single human trace."""
    model_final_correct = []
    for trace in model_traces:
        p_correct = np.where(
            trace[-1]["marked_heatmap"],
            (1 - epsilon) * trace[-1]["correct_heatmap"] + epsilon / 2,
            0.5,
        )  # Pure guess for unmarked cells
        model_final_correct.append(p_correct)
    model_final_correct = np.mean(model_final_correct, axis=0)

    final_state = human_trace[-1]

    h_correct = final_state["correct_heatmap"]

    ll = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] in [-1, -3, -4]:
                if h_correct[i][j] == 1:
                    ll += np.log(model_final_correct[i][j])
                else:
                    ll += np.log(1 - model_final_correct[i][j])

    return ll


def compute_state_likelihood(
    board, human_marked, human_correct, model_marked, model_correct
):
    """Compute likelihood of human data given model predictions (with epsilon already applied)."""
    ll = 0.0

    # For each cell
    for y in range(len(human_marked)):
        for x in range(len(human_marked[0])):
            if board[y][x] < 0:
                # If human marked this cell
                if human_marked[y, x] > 0:
                    # Add log probability of marking
                    ll += np.log(model_marked[y, x])

                    # If human was correct
                    if human_correct[y, x] > 0:
                        # Add log probability of being correct given marking
                        p_correct = model_correct[y, x] / model_marked[y, x]
                        ll += np.log(p_correct)
                    else:
                        # Add log probability of being incorrect given marking
                        p_incorrect = 1 - (model_correct[y, x] / model_marked[y, x])
                        ll += np.log(p_incorrect)
                else:
                    # Add log probability of not marking
                    ll += np.log(1 - model_marked[y, x])

    return ll


def get_participant_data(df):
    participant_data = {}

    # Group by prolific_id
    for prolific_id, participant_df in df.groupby("prolific_id"):
        # Store participant's data
        participant_data[prolific_id] = participant_df

    return participant_data


def logit(p):
    """Convert probability to log odds."""
    return np.log(p / (1 - p))


def inv_logit(x):
    """Convert log odds to probability."""
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    arr = np.arange(-20, 20)
    print(np.round(softmax(arr, tau=10), 2))
