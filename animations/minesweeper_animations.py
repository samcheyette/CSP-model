from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import json
import copy
import pandas as pd
from PIL import Image
import subprocess
import ast  # For safely parsing Python-style dictionaries
import re
import sys
import os
import shutil  # Add this to imports

# Add the parent directory to the system path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules from the parent directory
from grammar import *
from constraints import *
from utils.utils import *
from agent import *
# Game class not needed for rendering; avoid importing game package
from pathlib import Path

TILE_SIZE = 140
FONT_SIZE = 60
INSET = 2

FLAG_SIZE = int(TILE_SIZE * 0.5) 
SAFE_SIZE = int(TILE_SIZE * 0.5)
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" 
CELL_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
flag_img_path = os.path.join(_THIS_DIR, 'imgs', 'flag.png')
safe_img_path = os.path.join(_THIS_DIR, 'imgs', 'safe.png')

flag_img = Image.open(flag_img_path)
safe_img = Image.open(safe_img_path)

COLORS = {
    -1: (180, 180, 180),  # Unknown tile (gray)
    0: (220, 220, 220),   # Empty tile (light gray)
    1: (0, 0, 255),       # Blue for 1
    2: (0, 128, 0),       # Green for 2
    3: (255, 0, 0),       # Red for 3
    4: (0, 0, 128),       # Dark Blue for 4
    5: (128, 0, 0),       # Maroon for 5
    6: (0, 128, 128),     # Teal for 6
    7: (0, 0, 0),         # Black for 7
    8: (128, 128, 128),   # Gray for 8
}

# Colors for the 3D effect
LIGHT_EDGE = (255, 255, 255)  
DARK_EDGE = (100, 100, 100)   
OUTSIDE_WINDOW_COLOR = (105, 105, 105, 150) 

# Overlay colors for animations
CURRENT_CONSTRAINT_COLOR = (80, 80, 250, 40)
CURRENT_VARIABLE_COLOR = (255, 200, 50, 80)
PROPOSED_CONSTRAINT_COLOR = (80, 80, 250, 40)
PROPOSED_VARIABLE_COLOR = (255, 200, 50, 80)



def load_simulation_results(file_path='minesweeper_simulation_results.csv'):
    df = pd.read_csv(file_path)

    # Parse list-like columns stored as strings
    list_like_columns = [
        'unsolved_board',
        'solution_board',
        'board_state',
        'error_heatmap',
        'correct_heatmap',
        'constraint_heatmap',
        'variable_heatmap',
        'proposed_constraint_heatmap',
        'proposed_variable_heatmap',
        # Back-compat fields (may not exist in new format)
        'active_constraints_in_path',
        'constraints_in_path',
        'vars_in_path',
        'unassigned_in_path',
        'stimulus_idx',
    ]
    for col in list_like_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and pd.notna(x) else x)

    if 'proposal_variables' in df.columns:
        df['proposal_variables'] = df['proposal_variables'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and pd.notna(x) else x
        )

    return df




def render_game_state(game_state):
    rows = len(game_state)
    cols = len(game_state[0])

    # Flag and safe images
    flag_img = Image.open(flag_img_path).resize((FLAG_SIZE, FLAG_SIZE), Image.Resampling.LANCZOS)
    safe_img = Image.open(safe_img_path).resize((SAFE_SIZE, SAFE_SIZE), Image.Resampling.LANCZOS)
    
    separation = 40  # Space between each board
    title_height = 50  # Space for titles
    img_width = cols * TILE_SIZE 
    img_height = rows * TILE_SIZE + title_height
    img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    
    draw = ImageDraw.Draw(img, 'RGBA')
    
    try:
        font = ImageFont.truetype(CELL_FONT_PATH, FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()
        
    for y in range(rows):
        for x in range(cols):
            render_tile_on_board(draw, img, game_state[y][x], x * TILE_SIZE, y * TILE_SIZE , flag_img, safe_img, font)
    
    return img



def render_tile_on_board(draw, img, tile, x1, y1, flag_img, safe_img, font):
    """Renders a single tile on the board."""
    x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE
    
    # Convert tile to integer if it's a number
    if isinstance(tile, (int, float)) and tile > 0:
        tile = int(tile)
    
    if tile in [-1, -3, -4]:
        draw.rectangle([x1, y1, x2, y2], fill=(190, 190, 190), outline=(0, 0, 0))  # Base color for unrevealed tiles
        draw_3d_effect(draw, x1, y1, x2, y2)

        if tile == -3:
            flag_x = x1 + (TILE_SIZE - FLAG_SIZE) // 2
            flag_y = y1 + (TILE_SIZE - FLAG_SIZE) // 2
            img.paste(flag_img, (flag_x, flag_y), flag_img.convert("RGBA"))
        elif tile == -4:
            safe_x = x1 + (TILE_SIZE - SAFE_SIZE) // 2
            safe_y = y1 + (TILE_SIZE - SAFE_SIZE) // 2
            img.paste(safe_img, (safe_x, safe_y), safe_img.convert("RGBA"))
    else:
        color = COLORS.get(int(tile), (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255), outline=(0, 0, 0))
        if tile > 0:
            text = str(tile)  # Will now be an integer
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_x = x1 + (TILE_SIZE - (text_bbox[2] - text_bbox[0])) // 2
            text_y = y1 + (TILE_SIZE - (text_bbox[3] - text_bbox[1])) // 2 - (text_bbox[1] // 2)
            draw.text((text_x, text_y), text, fill=color, font=font)



def draw_3d_effect(draw, x1, y1, x2, y2):
    """Draws a 3D effect on unrevealed tiles."""
    draw.line([x1 + INSET, y1 + INSET, x2 - INSET, y1 + INSET], fill=LIGHT_EDGE, width=INSET)
    draw.line([x1 + INSET, y1 + INSET, x1 + INSET, y2 - INSET], fill=LIGHT_EDGE, width=INSET)
    draw.line([x1 + INSET, y2 - INSET, x2 - INSET, y2 - INSET], fill=DARK_EDGE, width=INSET)
    draw.line([x2 - INSET, y1 + INSET, x2 - INSET, y2 - INSET], fill=DARK_EDGE, width=INSET)


def highlight_square(draw, x, y, tile_size, color=(255, 255, 0, 80)):
    x1, y1 = x * tile_size, y * tile_size
    x2, y2 = x1 + tile_size, y1 + tile_size
    
    # Draw semi-transparent yellow highlight
    draw.rectangle([x1, y1, x2, y2], 
                  fill=color,  # Yellow with alpha
                  outline=(color[0], color[1], color[2], 255),  # Solid orange outline
                  width=3)


def _draw_label(draw, text):
    try:
        font = ImageFont.truetype(CELL_FONT_PATH, 28)
    except IOError:
        font = ImageFont.load_default()
    padding = 6
    text_bbox = draw.textbbox((0, 0), text, font=font)
    w = text_bbox[2] - text_bbox[0]
    h = text_bbox[3] - text_bbox[1]
    draw.rectangle([padding, padding, padding + w + 10, padding + h + 6], fill=(0, 0, 0, 100))
    draw.text((padding + 5, padding + 3), text, fill=(255, 255, 255, 255), font=font)


def process_simulations(simulation_results, output_base_dir='animations/imgs'):
    """
    Process simulation results and save images to the specified directory structure.
    
    Args:
        simulation_results: DataFrame containing simulation results
        output_base_dir: Base directory for output images
    """
    # Create base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Determine grouping based on available columns (new format uses unique_id, simulation_idx)
    max_sims = simulation_results['simulation_idx'].max() if 'simulation_idx' in simulation_results.columns else 0
    new_keys = ['capacity', 'R_init', 'ILtol_init', 'gamma', 'unique_id', 'simulation_idx']
    old_keys = ['memory_capacity', 'beta_IL', 'tau', 'search_budget', 'stimulus_idx', 'simulation_id']
    group_keys = [k for k in new_keys if k in simulation_results.columns]
    if not group_keys:
        group_keys = [k for k in old_keys if k in simulation_results.columns]
    grouped = simulation_results.groupby(group_keys) if group_keys else [(None, simulation_results)]
    
    output_dir_init = f"{output_base_dir}/minesweeper"
    if os.path.exists(output_dir_init):
        shutil.rmtree(output_dir_init)
    os.makedirs(output_dir_init, exist_ok=True)

    for _, group in grouped:
        # Extract identifiers/parameters for directory naming
        row0 = group.iloc[0]
        if 'board_id' in row0:
            uid = row0['board_id']
            sim_dir = f"{output_dir_init}/game_{uid}"
            param_parts = []
            for key in ['capacity', 'R_init', 'ILtol_init', 'gamma']:
                if key in row0:
                    param_parts.append(str(row0[key]))
            param_dir = f"params_{'_'.join(param_parts) if param_parts else 'default'}"
            simulation_idx = row0['simulation_idx'] if 'simulation_idx' in row0 else 0
        else:
            # Fallback to old format
            sim_dir = f"{output_dir_init}/game_{row0['stimulus_idx']}"
            param_dir = f"params_{row0['memory_capacity']}_{row0['beta_IL']}_{row0['tau']}_{row0['search_budget']}"
            simulation_idx = row0['simulation_id'] if 'simulation_id' in row0 else 0

        path = f"{sim_dir}/{param_dir}"
        if max_sims > 0:
            path = f"{path}/simulation_{simulation_idx}"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        print(f"Processing Board {sim_dir.split('_')[-1]}, Simulation {simulation_idx}")
        step_num = 0

        # Prefer explicit board_state; fallback to unsolved_board
        initial_board = group.iloc[0]['board_state'] if 'board_state' in group.columns else None
        # If missing or empty, fallback
        if initial_board is None or (isinstance(initial_board, float) and np.isnan(initial_board)) or (isinstance(initial_board, list) and len(initial_board) == 0):
            initial_board = group.iloc[0]['unsolved_board'] if 'unsolved_board' in group.columns else None
        initial_state = np.array(initial_board)

        initial_state = initial_state * (initial_state >= -1) + -1 * (initial_state < -1)
        initial_img = render_game_state(initial_state)
        img_path = os.path.join(path, f"{step_num:04d}.png")
        initial_img.save(img_path)
        step_num += 1

        for _, rec in group.iterrows():
            # Prefer board_state when available and non-empty; otherwise use unsolved_board
            board_state_val = rec['board_state'] if 'board_state' in rec else None
            game_state = None
            if isinstance(board_state_val, list) and board_state_val:
                game_state = board_state_val
            elif board_state_val is not None and not isinstance(board_state_val, float):
                game_state = board_state_val
            elif 'unsolved_board' in rec:
                game_state = rec['unsolved_board']

            current_var_hm = rec['variable_heatmap'] if 'variable_heatmap' in rec else []
            current_con_hm = rec['constraint_heatmap'] if 'constraint_heatmap' in rec else []
            proposed_var_hm = rec['proposed_variable_heatmap'] if 'proposed_variable_heatmap' in rec else None
            proposed_con_hm = rec['proposed_constraint_heatmap'] if 'proposed_constraint_heatmap' in rec else None
            proposal_decision = rec['proposal_decision'] if 'proposal_decision' in rec else None
            is_contradiction = rec['is_contradiction'] if 'is_contradiction' in rec else False

            # PROPOSAL frame
            img = render_game_state(game_state)
            draw = ImageDraw.Draw(img, 'RGBA')
            # Always draw current (subproblem) overlays first
            if isinstance(current_con_hm, list) and current_con_hm:
                for r in range(len(current_con_hm)):
                    for c in range(len(current_con_hm[r])):
                        if current_con_hm[r][c]:
                            highlight_square(draw, c, r, TILE_SIZE, color=CURRENT_CONSTRAINT_COLOR)
            if isinstance(current_var_hm, list) and current_var_hm:
                for r in range(len(current_var_hm)):
                    for c in range(len(current_var_hm[r])):
                        if current_var_hm[r][c]:
                            highlight_square(draw, c, r, TILE_SIZE, color=CURRENT_VARIABLE_COLOR)
            # Then draw proposed overlays on top
            if isinstance(proposed_con_hm, list) and proposed_con_hm:
                for r in range(len(proposed_con_hm)):
                    for c in range(len(proposed_con_hm[r])):
                        if proposed_con_hm[r][c] and not current_con_hm[r][c]:
                            highlight_square(draw, c, r, TILE_SIZE, color=CURRENT_CONSTRAINT_COLOR)
            if isinstance(proposed_var_hm, list) and proposed_var_hm:
                for r in range(len(proposed_var_hm)):
                    for c in range(len(proposed_var_hm[r])):
                        if proposed_var_hm[r][c] and not current_var_hm[r][c]:
                            highlight_square(draw, c, r, TILE_SIZE, color=CURRENT_VARIABLE_COLOR)
            img_path = os.path.join(path, f"{step_num:04d}.png")
            img.save(img_path)
            step_num += 1

            # DECISION frame: retain proposal if accepted, otherwise show only current overlays
            img2 = render_game_state(game_state)
            draw2 = ImageDraw.Draw(img2, 'RGBA')
            # Draw current overlays
            if isinstance(current_con_hm, list) and current_con_hm:
                for r in range(len(current_con_hm)):
                    for c in range(len(current_con_hm[r])):
                        if current_con_hm[r][c]:
                            highlight_square(draw2, c, r, TILE_SIZE, color=CURRENT_CONSTRAINT_COLOR)
            if isinstance(current_var_hm, list) and current_var_hm:
                for r in range(len(current_var_hm)):
                    for c in range(len(current_var_hm[r])):
                        if current_var_hm[r][c]:
                            highlight_square(draw2, c, r, TILE_SIZE, color=CURRENT_VARIABLE_COLOR)
            # If accepted, keep proposal highlight; if rejected, do not draw proposal
            if isinstance(proposal_decision, str) and proposal_decision.upper() == 'ACCEPT':
                if isinstance(proposed_con_hm, list) and proposed_con_hm:
                    for r in range(len(proposed_con_hm)):
                        for c in range(len(proposed_con_hm[r])):
                            if proposed_con_hm[r][c]:
                                highlight_square(draw2, c, r, TILE_SIZE, color=PROPOSED_CONSTRAINT_COLOR)
                if isinstance(proposed_var_hm, list) and proposed_var_hm:
                    for r in range(len(proposed_var_hm)):
                        for c in range(len(proposed_var_hm[r])):
                            if proposed_var_hm[r][c]:
                                highlight_square(draw2, c, r, TILE_SIZE, color=PROPOSED_VARIABLE_COLOR)
            img_path2 = os.path.join(path, f"{step_num:04d}.png")
            img2.save(img_path2)
            step_num += 1
    return output_base_dir  # Return the base directory for animation creation


def create_animations(base_dir='animations/imgs', fps=5):
    """
    Create MPEG animations from image sequences in the specified directory.
    
    Args:
        base_dir: Base directory containing the minesweeper folder
        fps: Frames per second for the animation
    """
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(os.path.dirname(base_dir), 'videos')

    if os.path.exists(videos_dir):
        shutil.rmtree(videos_dir)
    os.makedirs(videos_dir, exist_ok=True)

    # Path to minesweeper directory
    minesweeper_dir = os.path.join(base_dir, 'minesweeper')
    if not os.path.exists(minesweeper_dir):
        print(f"Directory not found: {minesweeper_dir}")
        return

    # Get all game directories
    game_dirs = [d for d in os.listdir(minesweeper_dir) 
                if os.path.isdir(os.path.join(minesweeper_dir, d)) and d.startswith('game_')]
    
    for game_dir in game_dirs:
        game_path = os.path.join(minesweeper_dir, game_dir)
        # game_dir already encodes the board id (unique_id in new format)
        
        # Create game video directory
        game_videos_dir = os.path.join(videos_dir, 'minesweeper', game_dir)
        os.makedirs(game_videos_dir, exist_ok=True)
        
        # Get all parameter directories
        param_dirs = [d for d in os.listdir(game_path) 
                     if os.path.isdir(os.path.join(game_path, d)) and d.startswith('params_')]
        
        for param_dir in param_dirs:
            param_path = os.path.join(game_path, param_dir)
            
            # Extract parameters from directory name (new format: params_<cap>_<R_init>_<ILtol_init>_<gamma>)
            params_parts = param_dir.split('_')
            if len(params_parts) < 5:
                print(f"Invalid parameter directory name: {param_dir}, skipping...")
                continue
            p1, p2, p3, p4 = params_parts[1:5]
            
            # Check if there are simulation subdirectories
            sim_dirs = [d for d in os.listdir(param_path) 
                       if os.path.isdir(os.path.join(param_path, d)) and d.startswith('simulation_')]
            
            if sim_dirs:  # If there are simulation subdirectories
                for sim_dir in sim_dirs:
                    sim_path = os.path.join(param_path, sim_dir)
                    simulation_idx = sim_dir.split('_')[1]
                    
                    # Generate video from this simulation's image sequence
                    _process_image_sequence(sim_path, game_videos_dir, 
                                          p1, p2, p3, p4,
                                          simulation_idx, fps)
            else:  # No simulation subdirectories - images are directly in param directory
                # Use "0" as default simulation index if there's only one simulation
                _process_image_sequence(param_path, game_videos_dir,
                                      p1, p2, p3, p4,
                                      "0", fps)


def _process_image_sequence(image_dir, videos_dir, memory_capacity, beta_IL, tau, search_budget, 
                           simulation_idx, fps):
    """Helper function to process a single image sequence and create a video"""
    
    # Define output video path
    output_video = os.path.join(videos_dir, 
                               f"params_{memory_capacity}_{beta_IL}_{tau}_{search_budget}.{simulation_idx}.mp4")
        
    # Check if there are images in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    if not image_files:
        print(f"No images found in {image_dir}, skipping...")
        return
    
    print(f"Creating animation for {image_dir}")
    
    # Use ffmpeg to create the animation
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', f"{image_dir}/%04d.png",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure dimensions are even
        output_video
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Animation saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating animation: {e}")
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create animations.")


if __name__ == "__main__":
    # Example usage
    # Point to the new CSV produced by run_simulations (relative to this file)
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir.parent / 'simulations' / 'minesweeper_animation.csv'
    simulation_results = load_simulation_results(str(csv_path))

    # Process simulations and save images under the animations/imgs folder next to this file
    imgs_base_dir = str(base_dir / 'imgs')

    output_dir = process_simulations(simulation_results, output_base_dir=imgs_base_dir)

    # Create animations from the saved images
    create_animations(imgs_base_dir, fps=3)