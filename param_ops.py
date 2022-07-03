import torch
from model import DEVICE

# Saves the parameters
def save_params(model, path):
  torch.save(model.state_dict(), path)
  print("Saved: ", "\n\n")
  for param_tensor in model.state_dict():
     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# save_params(NET, "saved_params")


# Loads the parameters on a given model
def load_params(model, params_path):
  model.load_state_dict(torch.load(params_path, map_location=torch.device(DEVICE)))

# load_params(NET, "saved_params")

# Resets the parameters
def reset_params(model):
  for layer in model.children():
    if hasattr(layer, "reset_parameters"):
      layer.reset_parameters()

# reset_model(NET)