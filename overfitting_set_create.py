import numpy as np
import pandas as pd
import argparse

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

from copy_dataset import write_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create data for overfitting test")
    parser.add_argument("path2metaframe", type=str, help="path to metaframe")
    parser.add_argument("path2folder_save", type=str, help="path to save file")
    parser.add_argument("nums", type=int, help="num per class")
    
    args = parser.parse_args()
    path2metaframe = args.path2metaframe 
    path2folder_save = args.path2folder_save
    nums = args.nums

    frame = pd.read_csv(path2metaframe)

    frames = frame.groupby("new_cls").sample(n=nums, random_state=RANDOM_STATE)['new_path2frame'].unique().tolist() 
    path2save = f"{path2folder_save}/overfitting.txt"
    write_files(path2save, frames)
    print(path2save)