import pandas as pd
import argparse

from mapping import bin_mapping, bin_mapping_labels


# ======================
# Скрипт для маппинга
# ======================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument("path2metaframe",type=str, help="path to metaframe.csv")
    parser.add_argument("new_path2metaframe",type=str, help="path to metaframe.csv")

    args = parser.parse_args()
    path2metaframe = args.path2metaframe
    new_path2metaframe = args.new_path2metaframe
    
    frame = pd.read_csv(path2metaframe)
    
    # --- Формируем новые относительные пути ---
    frame['new_path2frame'] = frame['path2frame'].apply(lambda x: f"./images/{x.split("/")[-1]}")
    # --- маппинг для новых лейблов ---
    frame['new_cls'] = frame['cls'].apply(lambda x: bin_mapping[x])
    frame['label'] = frame['new_cls'].apply(lambda x: bin_mapping_labels[x])

    frame.to_csv(new_path2metaframe, index=False)