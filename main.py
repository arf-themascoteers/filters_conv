from train import train
from test import test
import constants
import os
import torch


model = None
if os.path.exists(constants.DEFAULT_MODEL_PATH):
    model = torch.load(constants.DEFAULT_MODEL_PATH)
else:
   model = train()

test()
