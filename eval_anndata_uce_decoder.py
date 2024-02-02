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
from torch.utils.data import TensorDataset, DataLoader

import argparse
import pickle
import random


def test(model, test_loader, library_size, device):
    model.eval()
    with torch.no_grad():
        rates  = []
        dropouts = []
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            uce_embeds, categories = batch[0].to(device), batch[1].to(device)
            library_sizes = torch.ones_like(categories).float() * library_size
            
            
            
            px_rate, px_r, px_dropout = model(uce_embeds, categories, library_sizes)
            rates.append(px_rate.detach().cpu())
            dropouts.append(px_dropout.detach().cpu())

    return torch.vstack(rates).numpy(), torch.vstack(dropouts).numpy()
def create_dataset_from_anndata(adata, categorical_label=None, unique_categories=None):
    '''
        Given an AnnData, create a tensor dataset.
        
        Assume that .X contains unnormalized counts. 
        Assume .obsm["X_uce"] contains UCE embeddings.
        If categorical_label is passed, will add those labels as categories.
        
        Returns TensorDataset which provides batches of (counts, UCE embeddings, categories), 
            as well as a list of sorted categories from categorical_label
    '''
    uce_embeds = adata.obsm["X_uce"]
    
    if categorical_label is not None:
        categories = adata.obs[categorical_label].str.lower()
        cat_codes = pd.Categorical(categories, categories=unique_categories).codes
    else:
        unique_categories = [1]
        cat_codes = np.zeros(counts.shape[0])
    return TensorDataset(torch.tensor(uce_embeds), torch.tensor(cat_codes))




def evaluate(args):
    # Read Data and create dataloaders
    device = torch.device(args.device)
    adata = sc.read(args.adata_path)
    n_genes = args.n_genes
    
    if args.categorical_label is not None:
        f = open(args.category_names_path, "r")
        unique_categories = [s.lower() for s in f.read().splitlines()] # lowecase these
        f.close()
    else:
        unique_categories = ["none"]
        
    f = open(args.decoder_gene_names_path, "r")
    decoder_gene_names = f.read().splitlines()
    f.close()
    
    print(unique_categories)
    full_dataset = create_dataset_from_anndata(adata, args.categorical_label, unique_categories=unique_categories)
    
    test_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create Model and optimizer
    layer_sizes = [int(s.strip()) for s in args.layer_sizes.split(",")] # convert string csv of sizes to list
    if args.categorical_label is not None:
        categorical_variable_dim = len(unique_categories)
    else:
        categorical_variable_dim = None # no cat var
    print(categorical_variable_dim)
    model = UCEDecoderModel(n_genes=n_genes, layer_sizes=layer_sizes, uce_embedding_size=args.uce_embedding_size,
                                categorical_variable_dim=categorical_variable_dim, dropout=args.dropout)
    
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    
    print(model)
    device = torch.device(args.device)
    torch.cuda.set_device(args.device_num)
    print(f"Using Device {args.device_num}")
    model = model.to(device)

    rates, dropouts = test(model, test_loader, args.library_size, device)
    adata.obsm["decoded_rates"] = rates
    adata.obsm["decoded_dropouts"] = dropouts
    #adata.layers["decoded_gene_names"] = np.array(decoder_gene_names)
    adata.write(args.adata_save_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a UCE Decoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run Setup
    parser.add_argument('--adata_path', type=str,
                        help='Path to UCE embedded anndata to decode.')
    parser.add_argument('--adata_save_path', type=str,
                        help='Path to save anndata with decoded dropouts and rates.')
    parser.add_argument('--device', type=str,
                    help='Set GPU/CPU')
    parser.add_argument('--device_num', type=int,
                        help='Set GPU Number', default=0)
    parser.add_argument('--categorical_label', type=str,
                        help='Column in adata.obs with categorical values to add to UCE embedding.')
    parser.add_argument('--model_path', type=str,
                        help='Path to save model to.')
    parser.add_argument('--category_names_path', type=str,
                        help='Path to load category names from (txt file).')
    parser.add_argument('--decoder_gene_names_path', type=str,
                        help='Path to load decoder gene names from (txt file).')    
    
    # Model Setup
    parser.add_argument('--layer_sizes', type=str,
                        help='Size of model hidden layers. Should be a string of integers separated by commas.')
    parser.add_argument('--uce_embedding_size', type=int, default=1280,
                        help='Size of UCE embedding layer.')
    parser.add_argument('--n_genes', type=int, default=5000,
                        help='Number of decoded genes.')
    parser.add_argument('--batch_size', type=int,
                        help='Set batch size', default=4096)
    parser.add_argument('--library_size', type=int, default=25000,
                        help='Number of counts per cell.')
    
    
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
        adata_path="/lfs/local/0/yanay/new_tabula_HVG_uce_decoder.h5ad",
        adata_save_path="/lfs/local/0/yanay/new_tabula_HVG_uce_decoder_decoded.h5ad",
        library_size=25000,
    )

    args = parser.parse_args()
    torch.cuda.set_device(args.device_num)
    print(f"Using Device {args.device_num}")
    
    evaluate(args)