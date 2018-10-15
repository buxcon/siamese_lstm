import torch

from utils.data import SICK
from network import MaLSTM, Network

sick = SICK.load().generate()

model = MaLSTM(sick["max_seq_len"], sick["embeddings"])
loss_function = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adadelta(model.parameters())

network = Network(model, loss_function, optimizer, sick)
network.optimize()
network.evaluate()
