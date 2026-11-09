from ultralytics  import YOLO
import argparse

RANDOM_STATE = 42


def get_parameters_argparse():
    parser = argparse.ArgumentParser(description="train script")
    parser.add_argument("path2data_file", type=str, help="path to data.yaml file")
    parser.add_argument("imgsize", type=int, help="image size")
    parser.add_argument("project", type=str, help="project name")
    parser.add_argument("exp_name", type=str, help="experiment name")
    
    parser.add_argument("epochs", type=int, help="num epoch")
    parser.add_argument("patience", type=int, help="num patience")
    parser.add_argument("batch_size", type=int, help="batch_size for train")
    parser.add_argument("lr", type=int, help="learning rate") 
    
    return parser.parse_args()



if __name__ == "__main__":
    # --- parameters ---    
    args = get_parameters_argparse()
    path2data_file = args.path2data_file 
    imgsize = args.imgsize
    project = args.project
    exp_name = args.exp_name
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    lr = args.lr