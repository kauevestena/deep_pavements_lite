import os
import torch

# Use CUDA if available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

params_path = "params.json"

data_path = "data"


ext_in = ".jpg"
ext_out = ".png"

pathway_categories = ["roads", "sidewalks"]


clip_model_path = "deep_pavements_clip_model.pt"
