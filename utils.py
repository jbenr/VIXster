import os
import glob
import sys
import time
from pathlib import Path
from tabulate import tabulate, tabulate_formats
from dotenv import load_dotenv
import subprocess

load_dotenv()
AWS_ACCESS = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET_KEY")
AV_KEY = os.getenv("ALPHA_VANTAGE_KEY")


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} created.')
    # else:
    #     print(f'Directory {dir} already exists.')


def pdf(df):
    from pandas import DataFrame
    # def get_variable_name(var):
    #     for name, value in globals().items():
    #         if value is var:
    #             return name
    #     return None  # Variable not found
    # print(get_variable_name(df))
    try:
        print(tabulate(df, headers='keys', tablefmt=tabulate_formats[8]))
    except:
        try:
            df1 = DataFrame(df).T
            print(tabulate(df1, headers='keys', tablefmt=tabulate_formats[8]))
        except Exception as e: print(e)


def find_project_root(project_name):
    current_path = Path.cwd()
    while current_path != current_path.root:
        if current_path.name == project_name:
            return current_path.resolve()
        current_path = current_path.parent
    return None


def get_project_root(project_name='algotrader'):
    """
    Get the root directory of the current project.
    """
    return find_project_root(project_name)


def get_abs_path(relative_path):
    """
    Get the absolute path for a given relative path based on the project root directory.
    """
    return os.path.join(get_project_root(), relative_path)


def globber(directory_path, type):
    parquet_files = []
    for root, dirs, files in os.walk(directory_path):
        parquet_files.extend(glob.glob(os.path.join(root, f'*.{type}')))
    return parquet_files


def df_tail_glob(parquet_files):
    import pandas as pd
    dfs = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path).tail(1)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=False)
    combined_df.index.name = 'Trade_Date'
    combined_df.sort_values(by='Trade_Date', inplace=True)
    return combined_df


def oh_waiter(secs,desc=""):
    for i in range(secs, 0, -1):
        sys.stdout.write(f"\rWaiting {i} seconds...")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write(f"\rWaiting 0 seconds... Done {desc}!\n")


def find_repo_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            return str(parent)
    return None


def run_command(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"❗️ Error: {result.stderr}")


def git_push(commit_message):
    repo_path = find_repo_root()
    if not repo_path:
        print("❗️ No Git repository found!")
        return

    print(f"📂 Repo path detected: {repo_path}")

    run_command(["git", "add", "."], cwd=repo_path)
    run_command(["git", "commit", "-m", commit_message], cwd=repo_path)
    run_command(["git", "push", "origin", "main"], cwd=repo_path)


def main():
    git_push("Auto commit 🚀")


if __name__ == "__main__":
    main()
