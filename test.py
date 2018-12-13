import numpy as np
from common.model import *
import torch

model_pos = TemporalModel(17, 2, 17, 
    filter_widths=[3,3,3,3,3],
    causal = False,
    dropout=0.25,
    channels=1024,
    dense=False)


receptive_field = model_pos.receptive_field()

print("Receptive field : ", receptive_field)

pad = (receptive_field - 1)//2
causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print("Trainable parameter count : ", model_params)


if torch.cuda.is_available():
    model_pos = model_pos.cuda()

#Load Checkpoint
chk_filename = "./checkpoint/pretrained_h36m_cpn.bin"
print("loading checkpoint", chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc:storage)

print("This model was trained for {} epochs".format(checkpoint['epoch']))

model_pos.load_state_dict(checkpoint['model_pos'])