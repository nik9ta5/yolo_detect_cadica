import pandas as pd
import numpy as np 
import argparse

from sklearn.model_selection import StratifiedGroupKFold


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def write_files(
    path2file: str,
    frame_list: list[str]
):
    with open(path2file, "w", encoding="utf-8") as f:
        for item in frame_list:
            f.write(f"{item}\n")

def bg_frac_select(frame: pd.DataFrame, frac: float, label_bg: str, random_state: int):
    """Формирование фрейма с определенной частью bg в датасете"""
    frame_without_bg = frame[frame['new_cls'] != label_bg]
    frame_bg = frame[frame['new_cls'] == label_bg].sample(frac=frac, random_state=random_state)

    result_frame = pd.concat([frame_without_bg, frame_bg])

    return result_frame



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="split dataset")
    parser.add_argument("path2metaframe", type=str, help="path to meta_frame.csv")
    parser.add_argument("path2folder_save", type=str, help="folder for save files")
    parser.add_argument("num_split", type=int, help="number split")
    parser.add_argument("frac_bg", type=float, default=0.2, help="frac background")

    args = parser.parse_args()
    path2data = args.path2metaframe 
    path2folder_save = args.path2folder_save
    num_split = args.num_split
    frac_bg = args.frac_bg
    
    # === read frame ===
    frame = pd.read_csv(path2data)

    # --- splits ---
    sgkf = StratifiedGroupKFold(n_splits=num_split, shuffle=True, random_state=RANDOM_STATE)

    splits = []
    for k, (tr, va) in enumerate(sgkf.split(frame, y=frame["new_cls"], groups=frame["patient"])):
        splits.append((tr, va))

        tr_frame = frame.iloc[tr] # frame for train
        val_frame = frame.iloc[va] # frame for val

        tr_frame = bg_frac_select(tr_frame, frac_bg, '-', RANDOM_STATE)

        train_frames = tr_frame['new_path2frame'].unique().tolist()
        val_frames = val_frame['new_path2frame'].unique().tolist()

        path2folder_save_train = f"{path2folder_save}/train_fold_{k}.txt"
        path2folder_save_val = f"{path2folder_save}/val_fold_{k}.txt"

        write_files(path2folder_save_train, train_frames)
        write_files(path2folder_save_val, val_frames)

        print(f'===== fold: {k} ======')
        print(len(train_frames), len(val_frames))        
        print(f"path 2 train: {path2folder_save_train}")
        print(f"path 2 val: {path2folder_save_val}")
        print("patient intersection",
        set(tr_frame['patient'].unique().tolist()).intersection(set(val_frame['patient'].unique().tolist()))
        )
        print('\ntrain')
        print(tr_frame.groupby('new_cls').size() / tr_frame.shape[0] * 100)
        print('\nval')
        print(val_frame.groupby('new_cls').size() / val_frame.shape[0] * 100)
        print("")


        