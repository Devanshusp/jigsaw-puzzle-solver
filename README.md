# Jigsaw Puzzle Solver

This program is a computer vision-based system designed to solve jigsaw puzzles. It uses contour detection, color analysis, and geometric matching to join puzzle pieces. The program extracts edge features from puzzle piece images, computes similarity scores, and reconstructs the puzzle by matching corresponding edges.

## Try It Out

Follow these steps to run the Jigsaw Puzzle Solver:

### 1. Set up the environment

Follow instructions in `Environment` to create a virtual environment and download packages in `requirements.txt`.

### 2. Configure settings in `main.py`

```py
# Visualize every solving step
DISPLAY_STEPS = False

# Skip visualizing preprocessing solving steps for extracting pieces from initial image, preprocessing solving steps for each piece, and  piece matching steps respectively
SKIP_DISPLAY_PREPROCESS_PUZZLE = False
SKIP_DISPLAY_PREPROCESS_PIECES = False
SKIP_DISPLAY_SOLVE = False

# Display non-essential visualizing steps for deeper understanding of the process
DISPLAY_NON_ESSENTIAL_PREPROCESS_STEPS = False
DISPLAY_NON_ESSENTIAL_SOLVE_STEPS = False

# Skip preprocessing if already done before
SKIP_PREPROCESS = False

# Save preprocessing data in set folder
SAVE_PREPROCESS = True
SAVE_PREPROCESS_FOLDER = "images"

# Include random rotation for pieces after preprocessing and extracting puzzle pieces
ADD_RANDOM_ROTATION_TO_PIECES = True

# Select puzzles to solve by commenting or uncommenting entries in this list:
PUZZLES_TO_PROCESS = [
    "aurora12",
    "aurora30",
    "aurora63",
    "corn12",
    "corn30",
    "corn63",
    "path12",
    "path30",
    "path63",
    "red9",
    "red25",
    "red49",
    "younker12",
    "younker30",
    "younker63",
]
```

### 3. Run `main.py`

Check your terminal for detailed logs and progress updates as the program processes and solves the puzzle!

## BYOP: Bring Your Own Puzzle

Use this program to solve your custom puzzle! Follow these steps:

1. Navigate to [I'm a Puzzle](https://im-a-puzzle.com/make-puzzle)
2. Upload you image of choice, select settings and click `Play this puzzle`
3. Drag the pieces onto the blank white background and take a screenshot
4. Save the screenshot as a `.png`, and store it in the `/puzzles` folder
5. In `main.py`, add you png string name to the `PUZZLES_TO_PROCESS` list

## Environment

Follow the steps below to set up your environment for this project. Note that `uv` is completely optional, simply follow the alternative steps instead.

```bash
# 1. Create a virtual environment named .venv.
pip install uv
uv .venv
python -m venv .venv # alternatively without uv

# 2. Activate the virtual environment.
source .venv/bin/activate    # macOS and Linux
.venv\Scripts\activate       # Windows

# 3. Install dependencies.
uv pip install -r requirements.txt
pip install -r requirements.txt # alternatively without uv

# 4. Save any additional dependencies to requirements.txt.
uv pip freeze > requirements.txt
pip freeze > requirements.txt # alternatively without uv
```

## Project Architecture

The project is structured into several directories and files. Check out the architecture below.

```txt
jigsaw-puzzle-solver/
├── functions/
│   ├── preprocessing/
│   │   ├── piece_to_polygon.py -> Extract polygonal approximation
│   │   ├── puzzle_to_pieces.py -> Extract individual pieces
│   ├── solving/
│   │   ├── match_pieces.py -> Heuristics and match score
│   │   ├── potential_piece_matches.py -> Get potential matches
│   │   ├── solve_border.py -> Solves border using corners and edges
│   │   ├── solve_center.py -> Solves center using middle pieces
│   ├── utils/
│   │   ├── color_data.py -> Extract colors along an edge
│   │   ├── display_piece_corners.py -> Grid of classifications
│   │   ├── display_pieces.py -> Grid of pieces
│   │   ├── normalize_list.py -> Normalize range to [0,1]
│   │   ├── orient_piece_contours.py -> Rotate side, get contours
│   │   ├── piece_file_names.py -> Get piece names
│   │   ├── rotate_image.py -> Rotate an image
│   │   ├── save_data.py -> Save/retrieve JSON data
│   │   ├── visualize_solution.py -> Solution grid of pieces
│   ├── __init__.py
│   ├── core.py -> Functions to preprocess puzzle/pieces and solve
├── images/ -> Processed data and images
│   ├── pieces/ -> Piece data for each puzzle
│   │   ├── aurora12/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── aurora30/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── corn12/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── corn30/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── path12/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── path30/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── red9/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── red25/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── younker12/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   │   ├── younker30/
│   │   │   ├── data.json
│   │   │   ├── piece_#.png
│   ├── pieces_grid/
│   │   ├── ..._corners.png -> Grid of piece and edge labels
│   │   ├── ..._corner.png -> Corner pieces
│   │   ├── ..._border.png -> Edge pieces
│   │   ├── ..._non_border.png -> Middle pieces
│   │   ├── ..._border_#.png -> Generated borders
│   │   ├── ..._solution_#.png -> Generated solutions
├── puzzles/ -> Inputs and output solutions
│   ├── solutions/ -> Original images (not puzzle pieces)
│   │   ├── aurora.png
│   │   ├── corn.png
│   │   ├── path.png
│   │   ├── red.png
│   ├── aurora#.png -> Unsolved puzzles (pieces)
│   ├── corn#.png
│   ├── path#.png
│   ├── red#.png
│   ├── younker#.png
├── main.py
├── README.md <- You are here!
├── requirements.txt
```
