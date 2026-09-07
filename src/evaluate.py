"""This script evaluates two Python scripts: evaluate_stat.py and evaluate_judge.py.

It allows the user to run either or both scripts with optional command-line arguments.

Usage:
    python evaluate.py [OPTIONS]

Options:
    --run-evaluate      Run evaluate_stat.py only.
    --run-judge         Run evaluate_judge.py only.
    --evaluate-args     Arguments to pass to evaluate_stat.py.
    --judge-args        Arguments to pass to evaluate_judge.py.

"""

import argparse
import os
import subprocess


def run_evaluation(args: argparse.Namespace) -> None:
    """Run the evaluation scripts based on the provided command-line arguments.

    Args:
        args: The command-line arguments parsed by argparse.

    Raises:
        subprocess.CalledProcessError: If an error occurs while running the scripts.
        FileNotFoundError: If the required Python files are not found in the directory.
    """
    try:
        # Get the directory of the current script
        script_dir: str = os.path.dirname(os.path.abspath(__file__))

        # Construct paths to the scripts
        evaluate_stat_path: str = os.path.join(script_dir, "evaluate_stat.py")
        evaluate_judge_path: str = os.path.join(script_dir, "evaluate_judge.py")

        # Run evaluate_stat.py with arguments
        if args.run_evaluate or (not args.run_evaluate and not args.run_judge):
            print("Running evaluate_stat.py...")
            subprocess.run(
                ["python3", evaluate_stat_path] + args.evaluate_args, check=True
            )
            print("✅ evaluate_stat.py completed successfully!")

        # Run evaluate_judge.py with arguments
        if args.run_judge or (not args.run_evaluate and not args.run_judge):
            print("Running evaluate_judge.py...")
            subprocess.run(
                ["python3", evaluate_judge_path] + args.judge_args, check=True
            )
            print("✅ evaluate_judge.py completed successfully!")

    except subprocess.CalledProcessError as error:
        print(f"❌ An error occurred while running the scripts: {error}")
    except FileNotFoundError:
        print(
            "❌ Python file not found. Ensure the scripts are in the correct directory."
        )


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run evaluate_stat.py and evaluate_judge.py with optional arguments."
    )
    parser.add_argument(
        "--run-evaluate", action="store_true", help="Run evaluate.py only"
    )
    parser.add_argument(
        "--run-judge", action="store_true", help="Run evaluate_judge.py only"
    )
    parser.add_argument(
        "--evaluate-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments to pass to evaluate.py",
    )
    parser.add_argument(
        "--judge-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments to pass to evaluate_judge.py",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the evaluation scripts with the provided arguments
    run_evaluation(args)
