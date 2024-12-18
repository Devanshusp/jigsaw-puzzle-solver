"""
main.py - The main script to run the Jigsaw Puzzle Solver.
"""

from functions import preprocess_pieces, preprocess_puzzle, solve_puzzle

if __name__ == "__main__":
    # Visualize every solving step
    DISPLAY_STEPS = True

    # Skip visualizing preprocessing solving steps for extracting pieces from initial
    # image, preprocessing solving steps for each piece, and  piece matching steps
    # respectively
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
    SAVE_PREPROCESS_FOLDER = ".images"

    # Include random rotation for pieces after preprocessing and extracting puzzle
    # pieces
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

    for puzzle in PUZZLES_TO_PROCESS:
        if not SKIP_PREPROCESS:
            preprocess_puzzle(
                puzzle,
                add_random_rotation=ADD_RANDOM_ROTATION_TO_PIECES,
                save=SAVE_PREPROCESS,
                save_folder=SAVE_PREPROCESS_FOLDER,
                display_steps=DISPLAY_STEPS and not SKIP_DISPLAY_PREPROCESS_PUZZLE,
            )

            preprocess_pieces(
                puzzle,
                save=SAVE_PREPROCESS,
                save_folder=SAVE_PREPROCESS_FOLDER,
                display_steps=DISPLAY_STEPS and not SKIP_DISPLAY_PREPROCESS_PIECES,
                display_non_essential_steps=DISPLAY_NON_ESSENTIAL_PREPROCESS_STEPS,
            )

        solve_puzzle(
            puzzle,
            save=SAVE_PREPROCESS,
            save_folder=SAVE_PREPROCESS_FOLDER,
            display_steps=DISPLAY_STEPS and not SKIP_DISPLAY_SOLVE,
            display_non_essential_steps=DISPLAY_NON_ESSENTIAL_SOLVE_STEPS,
        )
