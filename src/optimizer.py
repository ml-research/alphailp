import random
import time
import torch
from collections import OrderedDict
from tqdm import tqdm

random.seed(a=1)



class WeightOptimizer(object):
    """
    optimizer of clause weights using gradient descent
    """

    def __init__(self, infer_module, train_idxs, labels, lr=1e-2, wd=0.0):
        self.IM = infer_module
        self.train_idxs = train_idxs
        self.labels = labels
        self.lr = lr
        self.wd = wd
        self.batch_size = 0.05
        self.bce_loss = torch.nn.BCELoss()
        self.set_optimizer(self.IM.Ws)

    def set_optimizer(self, params):
        """
        set torch optimizer
        """
        self.optimizer = torch.optim.RMSprop(
            params, lr=self.lr, weight_decay=self.wd)

    def minibatch(self, probs, labels):
        """
        get minibatch
        Inputs
        ------
        probs : torch.tensor((|train_idxs|, ))
            valuation vector of examples
            each dimension represents each example of the ilp problem
        labels : torch.tensor((|train_idxs|, ))
            label vector of examples
            each dimension represents each example of the ilp problem
        Returns
        -------
        probs_batch : torch.tensor((batch_size, ))
            valuation vector of examples selected in the minibatch
            each dimension represents each example of the ilp problem
        labels_batch : torch.tensor((batch_size, ))
            label vector of examples selected in the minibatch
            each dimension represents each example of the ilp problem
        """
        ls = list(range(len(probs)))
        batch_num = int(len(probs)*self.batch_size)
        ls_batch = torch.tensor(random.sample(ls, batch_num)).to(device)
        return probs[ls_batch], labels[ls_batch]

    def optimize_weights(self, epoch=500):
        """
        perform gradient descent to optimize clause weights
        Inputs
        ------
        epoch : int
            number of steps in gradient descent
        Returns
        -------
        IM : .infer.InferModule
            infer module that contains optimized weight vectors
        loss_list : List[float]
            list of training loss
        """
        best_loss = 9999
        best_iter = 0
        best_Ws = self.IM.Ws

        i = 0
        loss_list = []
        with tqdm(range(epoch)) as pbar:
            for i in pbar:
                valuation = self.IM.infer()
                probs = torch.gather(valuation, 0, self.train_idxs)

                probs_batch, labels_batch = self.minibatch(probs, self.labels)
                loss = self.bce_loss(probs_batch, labels_batch)

                loss_list.append(loss.item())
                if loss > 0:
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                i += 1
                pbar.set_postfix(OrderedDict(loss=loss.item()))
        return self.IM, loss_list