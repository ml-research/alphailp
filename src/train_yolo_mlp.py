from rtpt import RTPT
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from nsfr_utils import get_data_loader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime

import torch.multiprocessing as mp

from tqdm import tqdm
import matplotlib

import numpy as np
from percept import YOLOPerceptionModule

from valuation import *
from neural_utils import LogisticRegression, MLP

matplotlib.use("Agg")


torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--ap-log", type=int, default=10, help="Number of epochs before logging AP"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["twopairs", "threepairs", "red-triangle", "closeby",
                 "online", "online-pair", "nine-circles"],
        help="Use MNIST dataset",
    )
    parser.add_argument("--dataset-type", default="kandinsky", help="kandinsky or clevr")
    parser.add_argument(
        "--perception-model",
        choices=["yolo", "slotattention"],
        help="Choose yolo or slotattention for object recognition.",
    )
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument("--small-data", action="store_true", help="Use small training data.")
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multiple GPUs")

    parser.add_argument("--data-dir", type=str, help="Directory to data")
    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')
    parser.add_argument('--program-size', default=5, type=int,
                        help='number of clauses to compose logic programs')

    args = parser.parse_args()
    return args


def compute_acc(outputs, targets):
    # print(outputs.shape)
    # print(targets.shape)
    predicts = np.argmax(outputs, axis=1)
    return accuracy_score(targets, predicts)

def predict(net, predict_net, loader, device, th=None):
    predicted_list = []
    target_list = []
    count = 0
    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)
        x = net(imgs)
        predicted = predict_net(x.view(-1, 6 * 11)).squeeze()

        predicted_list.append(predicted.detach())
        target_list.append(target_set.detach())

    predicted = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.cat(target_list, dim=0).to(
        torch.int64).detach().cpu().numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set,  [m > thresh for m in predicted], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted])
        rec_score = recall_score(
            target_set,  [m > th for m in predicted], average=None)
        return accuracy, rec_score, th

def run(net, predict_net,  loader, optimizer, criterion, writer, args, device, train=False, epoch=0,  rtpt=None, max_obj_num=4):
    iters_per_epoch = len(loader)
    loss_list = []
    val_loss_list = []

    be = torch.nn.BCELoss()

    loss_sum = 0
    for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)
        # reset grad
        if train:
            optimizer.zero_grad()

        # infer and predict the target probability
        x = net(imgs)
        predicted = predict_net(x.view(-1, 6*11)).squeeze()

        # binary cross-entropy loss computation
        loss = be(predicted, target_set)
        loss_sum += loss.item()
        loss_list.append(loss.item())
        loss.backward()
        # update parameters for the step
        if optimizer != None and epoch > 0:
            optimizer.step()

    return loss_sum

def main(n):
    args = get_args()
    if not args.small_data:
        base = 'KP/'
    else:
        base = 'small_KP/'
    name = base + args.dataset + '/YOLOMLP_' + str(n)
    device = 'cuda:' + args.device
    writer = SummaryWriter(f"runs/{name}", purge_step=0)

    # get torch data loader
    train_loader, val_loader, test_loader = get_data_loader(args)

    start_epoch = 0
    #net = models.resnet50(pretrained=True)
    #net.to(device)
    #predict_net = LogisticRegression(input_dim=1000)
    #predict_net.to(device)

    net = YOLOPerceptionModule(e=6, d=11, device=device)
    predict_net = MLP(in_channels=6 * 11, out_channels=1)
    predict_net.to(device)
    # setting optimizer
    params = list(net.parameters()) + list(predict_net.parameters())
    ###params = list(predict_net.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, args.epochs, eta_min=0.00005)
    criterion = torch.nn.SmoothL1Loss()

    # Create RTPT object
    rtpt = RTPT(name_initials='HS',
                experiment_name=base + args.dataset + '/YOLOMLP_' + str(n), max_iterations=args.epochs)
    rtpt.start()

    # train loop
    loss_list = []
    for epoch in np.arange(start_epoch, args.epochs + start_epoch):
        # training step
        loss = run(
            net, predict_net, train_loader, optimizer, criterion, writer, args, device=device, train=True, epoch=epoch, rtpt=rtpt)
        writer.add_scalar("metric/train_loss", loss, global_step=epoch)
        #writer.add_scalar("metric/train_acc",
        #                          mean_acc, global_step=epoch)
        rtpt.step(subtitle=f"loss={loss:2.2f}")

        if epoch % 20 == 0:
            # validation split
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(net, predict_net, val_loader, device)
            writer.add_scalar("metric/val_acc",
                      acc_val, global_step=epoch)
            print("Predicting on training data set...")
            # training split
            acc, rec, th = predict(net, predict_net, train_loader, device, th=th_val)
            writer.add_scalar("metric/train_acc", acc, global_step=epoch)

            print("Predicting on test data set...")
            # test split
            acc_test, rec_test, th_test = predict(
                net, predict_net, test_loader, device, th=th_val)
            writer.add_scalar("metric/test_acc", acc_test, global_step=epoch)
            #def predict(net, predict_net, loader, device, th=None):
            print("training acc: ", acc, "threashold: ", th, "recall: ", rec)
            print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
            print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)


if __name__ == "__main__":
    for i in range(5):
        main(n=i)
