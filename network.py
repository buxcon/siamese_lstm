import torch
import copy
import time
import numpy

from enum import Enum
from math import ceil
from config import PathConfig, NetworkConfig
from utils.data import Data
from utils.common import ProgressBar


class MaLSTM(torch.nn.Module):
    @staticmethod
    def exponent_neg_manhattan_distance(left, right):
        pairwise_distance = torch.nn.PairwiseDistance(p=1).cuda()
        return torch.exp(-pairwise_distance(left, right)).cuda()

    def __init__(self, max_seq_len, embeddings):
        super(MaLSTM, self).__init__()

        self.max_seq_len = max_seq_len

        self.embedding_layer = torch.nn.Embedding(len(embeddings), NetworkConfig.EMBEDDING_DIM).cuda()
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = False

        self.left_hidden_layer = torch.nn.LSTM(input_size=NetworkConfig.EMBEDDING_DIM,
                                               hidden_size=NetworkConfig.HIDDEN_SIZE).cuda()
        self.right_hidden_layer = torch.nn.LSTM(input_size=NetworkConfig.EMBEDDING_DIM,
                                                hidden_size=NetworkConfig.HIDDEN_SIZE).cuda()

    def padding_input(self, rand_len_inputs):
        padded = []
        for rand_len_input in rand_len_inputs:
            padding_len = self.max_seq_len - len(rand_len_input)
            padding = [0 for _ in range(padding_len)]
            padded.append(padding + rand_len_input)
        return torch.LongTensor(padded).cuda()

    def forward(self, X_batch):
        left_input = [X_batch[i][0] for i in range(NetworkConfig.BATCH_SIZE)]
        right_input = [X_batch[i][1] for i in range(NetworkConfig.BATCH_SIZE)]

        encoded_left_input = self.embedding_layer(self.padding_input(left_input))
        encoded_right_input = self.embedding_layer(self.padding_input(right_input))

        _, (left_h, _) = self.left_hidden_layer(encoded_left_input.view(self.max_seq_len,
                                                                        NetworkConfig.BATCH_SIZE,
                                                                        NetworkConfig.EMBEDDING_DIM))
        _, (right_h, _) = self.right_hidden_layer(encoded_right_input.view(self.max_seq_len,
                                                                           NetworkConfig.BATCH_SIZE,
                                                                           NetworkConfig.EMBEDDING_DIM))

        return MaLSTM.exponent_neg_manhattan_distance(left_h[0], right_h[0])


class Network:
    class Phase(Enum):
        TRAIN = 1
        TRIAL = 2

        def __str__(self):
            if self is Network.Phase.TRAIN:
                return "train"
            else:
                return "validation"

    def __init__(self, model, loss_function, optimizer, data, lr_scheduler=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.data = data
        self.lr_scheduler = lr_scheduler

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, decay_every=8):
        lr = init_lr * (0.1 ** (epoch // decay_every))

        if epoch % decay_every == 0:
            print("Learning rate has been set to %f" % lr)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return optimizer

    def optimize(self, save_name=None):
        best_model = self.model
        min_loss = float("inf")

        total = 0
        for phase in Network.Phase:
            total += NetworkConfig.EPOCH * \
                     ceil(len(self.data[str(phase)][0]) / NetworkConfig.BATCH_SIZE) * NetworkConfig.BATCH_SIZE
        bar = ProgressBar(total, "Training model...")
        progress = 0

        for epoch in range(NetworkConfig.EPOCH):
            for phase in Network.Phase:
                if phase is Network.Phase.TRAIN:
                    if self.lr_scheduler:
                        self.model = self.lr_scheduler(self.model, NetworkConfig.EPOCH)
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0

                for batch in Data.batches(self.data[str(phase)]):
                    self.optimizer.zero_grad()

                    X, Y = batch[0], torch.Tensor(batch[1]).cuda()
                    output = self.model(X)
                    loss = self.loss_function(output, Y)

                    if phase is Network.Phase.TRAIN:
                        loss.backward()
                        self.optimizer.step()

                    progress += NetworkConfig.BATCH_SIZE
                    bar.refresh(progress)

                    running_loss += loss.data.cpu().numpy()

                epoch_loss = running_loss / len(self.data[str(phase)])

                if phase is Network.Phase.TRIAL and epoch_loss < min_loss:
                    best_model = copy.deepcopy(self.model)
                    min_loss = epoch_loss

        bar.finish("Model has been optimized in %d epochs with %f as its minimal loss."
                   % (NetworkConfig.EPOCH, min_loss))

        if save_name is None:
            save_name = time.strftime("%Y%m%d_%H%M%S")
        save_path = "%s%s.mdl" % (PathConfig.MODEL_PATH, save_name)
        print("Saving model into %s..." % save_path)
        torch.save(best_model, save_path)
        print("Saved.\n")

        self.model = best_model

    def evaluate(self, save_name=None):
        if save_name is None:
            model = self.model.cuda()
        else:
            save_path = "%s%s.mdl" % (PathConfig.MODEL_PATH, save_name)
            print("Loading model from %s..." % save_path)
            model = torch.load(save_path).cuda()
            print("Loaded.\n")

        total = ceil(len(self.data["test"][0]) / NetworkConfig.BATCH_SIZE) * NetworkConfig.BATCH_SIZE
        correct_count = 0

        bar = ProgressBar(total, "Testing model...")
        progress = 0

        for batch in Data.batches(self.data["test"]):
            X, Y = batch[0], batch[1]
            output = model(X)

            output = numpy.round(output.cpu().detach().numpy() * 5)
            Y = numpy.round(numpy.array(Y) * 5)
            correct_count += numpy.sum(output == Y)

            progress += NetworkConfig.BATCH_SIZE
            bar.refresh(progress)

        bar.finish("Accuracy: %f" % (correct_count / total))
