import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from generation import generate_aptamers_for_protein
from feature import *
from xgboost import XGBClassifier
import pickle
import os

    
if __name__ == "__main__":
    
    #print("CREB3")
    #generate_aptamers_for_protein(2000, 20, 120, 0.6, "./dataset/sota/CREB3/CREB3_", "./dataset/sota/CREB3/CREB3.fasta", "./dataset/sota/CREB3/CREB3_SS.fas")
    
    #print("GSX1")
    #generate_aptamers_for_protein(2000, 20, 120, 0.6, "./dataset/sota/GSX1/GSX1_", "./dataset/sota/GSX1/GSX1.fasta", "./dataset/sota/GSX1/GSX1_SS.fas")
    
    #print("MESP1")
    #generate_aptamers_for_protein(2000, 20, 120, 0.5, "./dataset/sota/MESP1/MESP1_", "./dataset/sota/MESP1/MESP1.fasta", "./dataset/sota/MESP1/MESP1_SS.fas")
    
    #print("SOX18")
    #generate_aptamers_for_protein(2000, 20, 120, 0.5, "./dataset/sota/SOX18/SOX18_", "./dataset/sota/SOX18/SOX18.fasta", "./dataset/sota/SOX18/SOX18_SS.fas")
    
    #print("TGIF2LX")
    #generate_aptamers_for_protein(2000, 20, 120, 0.3, "./dataset/sota/TGIF2LX/TGIF2LX_", "./dataset/sota/TGIF2LX/TGIF2LX.fasta", "./dataset/sota/TGIF2LX/TGIF2LX_SS.fas")
    
    