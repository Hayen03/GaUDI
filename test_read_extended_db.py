
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from data.aromatic_dataloader import create_data_loaders
from utils.args_edm import Args_EDM
import re
from utils.extend_df import extend_df, DFKeys, develop_df, gen_xyz
import matplotlib.pyplot as plt

def draw_mol(coords, edges, smiles=None):
    X = [c[0] for c in coords]
    Y = [c[1] for c in coords]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X, Y, color="black")
    for i, j in edges:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], color='black', linewidth=2)
    ax.set_aspect('equal')
    if smiles is not None:
        ax.set_title(smiles)
    plt.show()
    return

def draw_transmission_curve(data, minx, maxx):
    X = np.linspace(minx, maxx, len(data))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(X, data, color="black")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Transmission")
    ax.set_title("Transmission Curve")
    plt.show()

def main(args):
    train_loader, val_loader, test_loader = create_data_loaders(args)
    l = train_loader.dataset.df.shape[0]
    
    for _ in range(0):
        row = train_loader.dataset.df.iloc[random.randint(0, l)]
        #print(f"Smile: {row["smiles"]}")
        adj_matrix = row[DFKeys.ADJ_MATRIX]
        #print(adj_matrix)
        coords, edges = gen_xyz(adj_matrix)
        #print(list(map(lambda c: c[:], coords)))
        #print(edges)
        draw_mol(coords, edges, row["smiles"])
        draw_transmission_curve(row[DFKeys.TRANSMISSIONS], row[DFKeys.MIN_TRANS_X], row[DFKeys.MAX_TRANS_X])
        
    print(train_loader.dataset[random.randint(0, l)])
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

    #print(args.exp_dir)
    #print("Args:", args)
    
    
    main(args)