"""
main.py - The main script to run the Jigsaw Puzzle Solver.
"""

from functions import preprocess_pieces, preprocess_puzzle, solve_puzzle

if __name__ == "__main__":
    # set settings before run
    DISPLAY_STEPS = False
    SKIP_DISPLAY_PREPROCESS_PUZZLE = False
    SKIP_DISPLAY_PREPROCESS_PIECES = False
    SKIP_DISPLAY_SOLVE = False
    SKIP_PREPROCESS = False
    SAVE_PREPROCESS = True
    SAVE_PREPROCESS_FOLDER = ".images2"
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
        if not SKIP_PREPROCESS:
            preprocess_puzzle(
                puzzle,
                save=SAVE_PREPROCESS,
                save_folder=SAVE_PREPROCESS_FOLDER,
                display_steps=DISPLAY_STEPS and not SKIP_DISPLAY_PREPROCESS_PUZZLE,
            )

            preprocess_pieces(
                puzzle,
                save=SAVE_PREPROCESS,
                save_folder=SAVE_PREPROCESS_FOLDER,
                display_steps=DISPLAY_STEPS and not SKIP_DISPLAY_PREPROCESS_PIECES,
            )

        solve_puzzle(
            puzzle,
            save=SAVE_PREPROCESS,
            save_folder=SAVE_PREPROCESS_FOLDER,
            display_steps=DISPLAY_STEPS and not SKIP_DISPLAY_SOLVE,
        )
