"""
main.py - The main script to run the Jigsaw Puzzle Solver.
"""

from functions import preprocess_puzzle

if __name__ == "__main__":
    # set settings before run
    SAVE_PREPROCESS = False
    SAVE_PREPROCESS_FOLDER = ".images"
    PUZZLES_TO_PROCESS = [
        "aurora12",
        # "aurora30",
        # "aurora63",
        # "corn12",
        # "corn30",
        # "corn63",
        # "path12",
        # "path30",
        # "path63",
        # "red9",
        # "red25",
        # "red49",
        # "younker12",
        # "younker30",
        # "younker63",
    ]

    for puzzle in PUZZLES_TO_PROCESS:
        preprocess_puzzle(
            puzzle, save=SAVE_PREPROCESS, save_folder=SAVE_PREPROCESS_FOLDER
        )
