from re import A
import pandas as pd
import os
import numpy as np
from itertools import combinations, product, chain
from functools import reduce

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.insert(0, parent_dir)  
from collections import defaultdict
from typing import Dict, Set, Iterator, Tuple, List, Any, Union
from grammar import Expression, Variable, Sum, Number, GreaterThan, LessThan, VariableState
from math import comb
from utils.utils import *
from utils.assignment_utils import *
import random
from collections import defaultdict
import numpy as np
from constraints import *
from agent import Agent
import csv
from special_constraints.sudoku_constraints import UniquenessConstraint


# pip install networkx matplotlib
import networkx as nx
import matplotlib.pyplot as plt


def var_id(v):      # unique, stable id
    return getattr(v, "name", str(v))

def _format_var_label(name: str):
    """Return a mathtext label for a variable name with numeric subscript.
    Examples:
      v0     -> $v_{0}$
      v_0_0  -> $v_{00}$
      v_3_12 -> $v_{312}$
      other  -> $other$ (underscores escaped)
    """
    try:
        if name.startswith("v_"):
            parts = name.split("_")[1:]
            digits = "".join(parts) if all(p.isdigit() for p in parts) else None
            if digits:
                return f"$v_{{{digits}}}$"
        if name.startswith("v") and name[1:].isdigit():
            return f"$v_{{{name[1:]}}}$"
        # generic fallback, escape underscores
        safe = name.replace("_", r"\_")
        return f"${safe}$"
    except Exception:
        return name


def var_label(v):   # human label on the node
    return _format_var_label(getattr(v, "name", str(v)))

def cons_id(c, idx=None):
    # use an explicit id if your class has one; else generate
    base = getattr(c, "name", c.__class__.__name__)
    if idx is not None:
        base = f"{base}_{idx}"
    return base

def cons_label(c):
    # make a readable label; specialize per class if you like
    if c.__class__.__name__ == "EqualityConstraint":
        # assume .variables (set of Variables) and .k (the sum target)
        k = getattr(c, "k", getattr(c, "target", "?"))
        return f"Σ={k}"
    elif c.__class__.__name__ == "InequalityConstraint":
        return f"Σ>{c.target}" if c.greater_than else f"Σ<{c.target}"
    elif c.__class__.__name__ == "UniquenessConstraint":
        #not equal to
        return f"≠"

    return c.__class__.__name__

def cons_variables(c):
    # iterable of Variables that the constraint touches
    return list(getattr(c, "variables", []))

# ---- graph construction ----

def build_csp_graph(variables, constraints):
    """
    variables: iterable[Variable]
    constraints: iterable[Constraint]
    Returns a NetworkX Graph with bipartite= {'var','con'} node attr.
    """
    G = nx.Graph()
    # add variables
    for v in variables:
        vid = var_id(v)
        G.add_node(vid, kind="var", obj=v, label=var_label(v))


    for i, c in enumerate(constraints):
        cid = cons_id(c, i)
        G.add_node(cid, kind="con", obj=c, label=cons_label(c))
        for v in cons_variables(c):
            G.add_edge(var_id(v), cid)

    return G

# ---- layout & drawing ----

def _bipartite_positions(G, comp_padding=2.5):
    """
    Positions variables (left) and constraints (right) per connected component
    so small components don’t overlap. Returns a dict: node -> (x,y).
    """
    pos = {}
    comps = list(nx.connected_components(G))
    x_offset = 0.0

    for comp in comps:
        H = G.subgraph(comp)
        left = [n for n in H if H.nodes[n]["kind"] == "var"]
        right = [n for n in H if H.nodes[n]["kind"] == "con"]

        # vertical stacking within each side
        left_y  = {n: i for i, n in enumerate(sorted(left))}
        right_y = {n: i for i, n in enumerate(sorted(right))}
        height = max(len(left), len(right)) - 1 if max(len(left), len(right))>0 else 1

        for n in left:
            pos[n] = (x_offset + 0.0, (left_y[n] - height/2))
        for n in right:
            pos[n] = (x_offset + 1.6, (right_y[n] - height/2))

        # bump x for the next component
        x_offset += 1.6 + comp_padding

    return pos

def _board_based_positions(G, constraint_offset=0.30, scale=1.0):
    """If nodes have row/col (board) coordinates, build a 2D layout.

    - Variables placed at (col, row)
    - Constraints (if have row/col) nudged by constraint_offset so they don't overlap variables
    Returns: dict node -> (x,y) or empty dict if coords missing
    """
    pos = {}
    have_any_coords = False
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for n, d in G.nodes(data=True):
        obj = d.get("obj", None)
        if obj is None:
            continue
        x = getattr(obj, "col", None)
        y = getattr(obj, "row", None)
        if x is None or y is None:
            # Some constraints/variables may not have embedded board coords
            continue
        have_any_coords = True
        if d.get("kind") == "con":
            x = x + constraint_offset
        pos[n] = (float(x), -float(y))
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)

    if not have_any_coords:
        return {}

    # Normalize and scale
    dx = max(1e-6, max_x - min_x)
    dy = max(1e-6, max_y - min_y)
    for n in pos:
        x, y = pos[n]
        x = (x - min_x) / dx
        y = (y - min_y) / dy
        pos[n] = (x * scale, y * scale)
    return pos


def _build_augmented_graph(G, w_bipartite=1.0):
    """Augment bipartite graph with intra-layer similarity edges so related
    constraints/variables cluster in 2D even without board coords.

    """
    H = nx.Graph()
    # copy nodes
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    # Precompute degrees in original graph
    #deg = dict(G.degree())
    # base bipartite edges (boost degree-1 pairs to keep them tight)
    for u, v in G.edges():
        base = w_bipartite
        # if deg.get(u, 0) <= 1 or deg.get(v, 0) <= 1:
        #     base += degree1_boost

        #print(u, v, H.get_edge_data(u, v, {}).get('weight', 0), base)
        H.add_edge(u, v, weight=H.get_edge_data(u, v, {}).get('weight', 0) + base)


    return H


def _paired_initial_positions(G, jitter=0.0, radius=2.0):
    """Create an initial 2D layout that pairs degree-1 variable/constraint neighbors.

    - Degree-1 var placed near its sole constraint
    - Degree-1 constraint placed near its sole variable
    - Others scattered on a circle
    """
    pos = {}
    rng = np.random.default_rng(42)

    # Seed coordinates for constraints arbitrarily on a circle
    nodes = list(G.nodes())
    cons = [n for n, d in G.nodes(data=True) if d.get("kind") == "con"]

    # Place constraints on a circle for a coarse scaffold
    for i, c in enumerate(cons):
        theta = 2 * np.pi * (i / max(1, len(cons)))
        pos[c] = (radius * np.cos(theta), radius * np.sin(theta))


    # Scatter remaining unplaced nodes on a larger circle
    for i, n in enumerate(nodes):
        theta = 2 * np.pi * (i / max(1, len(nodes)))
        r = radius * 1.5
        pos[n] = (r * np.cos(theta) + rng.normal(0, jitter), r * np.sin(theta) + rng.normal(0, jitter))
    return pos


def draw_csp_graph(G, ax=None, with_labels=True, node_size=600, lw=1.2, use_board_coords=True):
    """
    Blue circles = variables; gray squares = constraints.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Try a 2D board-aware layout first (if row/col exists)
    pos = {}
    if use_board_coords:
        pos = _board_based_positions(G, constraint_offset=0.50, scale=3.0)
        if pos:
            # Optionally refine with a light spring pass to spread within local neighborhoods
            try:
                pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.5, iterations=40)
            except Exception:
                pass
        else:
            # Fall back to augmented spring layout seeded by bipartite positions
            init_pos = _bipartite_positions(G)
            H = _build_augmented_graph(G, w_bipartite=1.0)
            #try:
            pos = nx.spring_layout(H, pos=init_pos, weight='weight', k=0.6, iterations=200)
            # except Exception:
            #pos = nx.kamada_kawai_layout(H, weight='weight')
    else:
        # Pure graph layout: augment with intra-layer similarities so related
        # constraints/variables cluster together
        init_pos = _paired_initial_positions(G)
        H = _build_augmented_graph(G, w_bipartite=1.0)
        # Spring layout with initial positions on the original node set
        pos = nx.spring_layout(H, pos=init_pos, weight='weight', k=0.7, iterations=300)

    vars_ = [n for n, d in G.nodes(data=True) if d["kind"] == "var"]
    cons_ = [n for n, d in G.nodes(data=True) if d["kind"] == "con"]

    # edges (draw actual CSP edges for clarity)
    nx.draw_networkx_edges(G, pos, ax=ax, width=max(1.2, lw), alpha=0.85)

    # variable nodes (circles)
    nx.draw_networkx_nodes(
        G, pos, nodelist=vars_, node_color="#4ea3ff",
        node_shape="o", edgecolors="black", linewidths=0.5, node_size=int(node_size*0.6), ax=ax
    )
    # constraint nodes (squares)
    nx.draw_networkx_nodes(
        G, pos, nodelist=cons_, node_color="#bfbfbf",
        node_shape="s", edgecolors="black", linewidths=0.5, node_size=int(node_size*0.5), ax=ax
    )

    if with_labels:
        # Attach labels at node centers, smaller font, with light background for readability
        for n in vars_:
            x, y = pos[n]
            txt = G.nodes[n].get("label", n)
            ax.text(x, y, txt, fontsize=8, ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
        for n in cons_:
            x, y = pos[n]
            txt = G.nodes[n].get("label", n)
            ax.text(x, y, txt, fontsize=7, ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.6))

    ax.axis("off")
    return ax

# ---- convenience wrapper ----

def visualize_csp(*args, use_board_coords=True):
    """Visualize a CSP.

    Usage:
      visualize_csp(constraints)
      visualize_csp(variables, constraints)
    """
    if len(args) == 1:
        constraints = args[0]
        variables = set().union(*[c.get_variables() for c in constraints])
    elif len(args) == 2:
        variables, constraints = args
    else:
        raise ValueError("visualize_csp expects (constraints) or (variables, constraints)")

    G = build_csp_graph(variables, constraints)
    draw_csp_graph(G, use_board_coords=use_board_coords)
    plt.show()
    return G


if __name__ == "__main__":
    # constraints, true_assignments = generate_random_constraints(n_variables=10, n_constraints=6, p_inequality=0.0, avg_size=2, sd_size=1)
    # for c in constraints:
    #     print(c)
        
    # print()

    #variables = set().union(*[c.get_variables() for c in constraints])

    from csp_games import minesweeper as minesweeper_game
    from csp_games.minesweeper import board_to_constraints
    file = "minesweeper_7x7_hard"

    path = os.path.join(parent_dir, 'csp_games', 'puzzles')
    data = minesweeper_game.load_boards(f"{file}", filepath=path)
    unsolved_boards = [np.array(d["game_state"]) for d in data]
    solved_boards = [np.array(d["game_board"]) for d in data]
    board_args_list = []

    unsolved_board = unsolved_boards[0]
    solved_board = solved_boards[0]


    constraints = board_to_constraints(unsolved_board)
    variables = set().union(*[c.get_variables() for c in constraints])
    G = build_csp_graph(variables, constraints)
    draw_csp_graph(G)
    plt.show()