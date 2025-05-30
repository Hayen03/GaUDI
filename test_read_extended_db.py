
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from utils.args_edm import Args_EDM
import re
from utils.extend_df import extend_df, DFKeys, develop_df

def main(args):
    #train_loader, _, _ = create_data_loaders(args)
    #print(train_loader)
	df = pd.read_csv(r"d:\Documents\dev\generator-main_v2\DATASETS\MODIFIED_COMPAS-1D.csv")
    #print(df)
	row = df.loc[0]
	extend_df(df)
	df = develop_df(df)
	print(df.columns)
	print(df[[DFKeys.CONTACTS, DFKeys.TRANSMISSIONS]])
	return

if __name__ == "__main__":
	# set seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load arguments and save in the experiment directory
    args = Args_EDM().parse_args()
    args.exp_dir = f"{args.save_dir}/{args.name}"

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    with open(args.exp_dir + "/args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(args.exp_dir)
    print("Args:", args)
    
    
    main(args)