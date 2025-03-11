import pandas as pd
import torch
from torch.utils.data import DataLoader

file_path = 'data/features/final_df.csv'
df = pd.read_csv(file_path)

print(df.head(50))