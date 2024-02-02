import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import pandas as pd

import torch
import torch.optim as optim
import numpy as np


from model import UCEDecoderModel
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split

import argparse
import pickle
import random


def train(model, train_loader, optimizer, epoch, device):
    '''Train MODEL on data from TRAIN_LOADER optimize using OPTIMIZER for epoch number EPOCH'''
    model.train()
    running_losses = []
    for batch_idx, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        counts, uce_embeds, categories = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        library_size = counts.sum(1)
        px_rate, px_r, px_dropout = model(uce_embeds, categories, library_size)
        loss = model.get_reconstruction_loss(counts, px_rate, px_r, px_dropout)
        loss.backward()
        optimizer.step()
        running_losses.append(loss.item())
        
        batch_idx += 1
        if batch_idx % 25 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, np.round(np.mean(running_losses), 1)))
            running_losses = []
    print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, np.round(np.mean(running_losses), 1)))

def test(model, test_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        batch_losses = []
        batch_sizes = []
        for batch_idx, batch in enumerate(test_loader):
            counts, uce_embeds, categories = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            library_size = counts.sum(1)
            px_rate, px_r, px_dropout = model(uce_embeds, categories, library_size)
            loss = model.get_reconstruction_loss(counts, px_rate, px_r, px_dropout)
            batch_losses.append(loss.item())
            batch_sizes.append(counts.shape[0])
            
    test_loss = np.average(batch_losses, weights=batch_sizes)
    print("Epoch {} Test Loss = {}".format(epoch, np.round(test_loss, 3)))


def create_dataset_from_anndata(adata, categorical_label=None):
    '''
        Given an AnnData, create a tensor dataset.
        
        Assume that .X contains unnormalized counts. 
        Assume .obsm["X_uce"] contains UCE embeddings.
        If categorical_label is passed, will add those labels as categories.
        
        Returns TensorDataset which provides batches of (counts, UCE embeddings, categories), 
            as well as a list of sorted categories from categorical_label
    '''

    counts = adata.X
    uce_embeds = adata.obsm["X_uce"]
    
    if categorical_label is not None:
        categories = adata.obs[categorical_label]
        
        unique_categories = sorted(np.unique(categories))
        
        cat_codes = pd.Categorical(categories, categories=unique_categories).codes
    else:
        unique_categories = ["none"]
        cat_codes = np.zeros(counts.shape[0])
    return TensorDataset(torch.tensor(counts), torch.tensor(uce_embeds), torch.tensor(cat_codes)), unique_categories




def trainer(args):
    # Read Data and create dataloaders
    device = torch.device(args.device)
    adata = sc.read(args.adata_path)
    
    decoder_gene_names = adata.var_names
    n_genes = adata.X.shape[1]
    full_dataset, unique_categories = create_dataset_from_anndata(adata, args.categorical_label)
    with open(args.category_names_path, "w+") as f:
        f.write("\n".join(unique_categories)) # save in different lines
    with open(args.decoder_gene_names_path, "w+") as f:
        f.write("\n".join(decoder_gene_names))
    train_size = int(0.95 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create Model and optimizer
    layer_sizes = [int(s.strip()) for s in args.layer_sizes.split(",")] # convert string csv of sizes to list
    if args.categorical_label is not None:
        categorical_variable_dim = len(unique_categories)
    else:
        categorical_variable_dim = None # no cat var
    
    model = UCEDecoderModel(n_genes=n_genes, layer_sizes=layer_sizes, uce_embedding_size=args.uce_embedding_size,
                                categorical_variable_dim=categorical_variable_dim, dropout=args.dropout)
    device = torch.device(args.device)
    torch.cuda.set_device(args.device_num)
    print(f"Using Device {args.device_num}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # TRAIN the decoder
    print("*****STARTING TRAINING*****")
    for epoch in range(1, args.num_epochs + 1):
        train(model, train_loader, optimizer, epoch, device)
        test(model, test_loader, epoch, device)
    
    print("*****Wrote model to *****")
    print(args.model_path)
    print(args.category_names_path)
    torch.save(model.state_dict(), args.model_path) # save after finishing training
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a UCE Decoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run Setup
    parser.add_argument('--adata_path', type=str,
                        help='Path to UCE embedded anndata.')
    parser.add_argument('--device', type=str,
                    help='Set GPU/CPU')
    parser.add_argument('--device_num', type=int,
                        help='Set GPU Number', default=0)
    parser.add_argument('--seed', type=int,
                        help='Init Seed', default=0)
    parser.add_argument('--categorical_label', type=str,
                        help='Column in adata.obs with categorical values to add to UCE embedding.')
    parser.add_argument('--model_path', type=str,
                        help='Path to save model to.')
    parser.add_argument('--category_names_path', type=str,
                        help='Path to save category names to.')
    parser.add_argument('--decoder_gene_names_path', type=str,
                        help='Path to save decoder gene names to (txt file).')    
    # Model Setup
    parser.add_argument('--layer_sizes', type=str,
                        help='Size of model hidden layers. Should be a string of integers separated by commas.')
    parser.add_argument('--uce_embedding_size', type=int, default=1280,
                        help='Size of UCE embedding layer.')
    parser.add_argument('--n_genes', type=int, default=5000,
                        help='Number of decoded genes.')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--dropot', type=float, default=0.05,
                        help='Dropout chance')
    parser.add_argument('--batch_size', type=int,
                        help='Set batch size', default=4096)
    
    
    
    # Defaults
    parser.set_defaults(
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_num=0,
        categorical_label=None,
        uce_embedding_size=1280,
        layer_sizes="1024,1024",
        num_epochs=5,
        dropout=0.05,
        batch_size=4096,
        model_path="uce_decoder_tabula_model.pt",
        category_names_path="tabula_categories.txt",
        decoder_gene_names_path="tabula_decoder_gene_names.txt",
        adata_path="/lfs/local/0/yanay/new_tabula_HVG_uce_decoder.h5ad"
    )

    args = parser.parse_args()
    torch.cuda.set_device(args.device_num)
    print(f"Using Device {args.device_num}")
    # Numpy seed
    np.random.seed(args.seed)
    # Torch Seed
    torch.manual_seed(args.seed)
    # Default random seed
    random.seed(args.seed)    
    print(f"Set seed to {args.seed}")
    trainer(args)