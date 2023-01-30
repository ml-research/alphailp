import argparse
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, accuracy_score
from rtpt import RTPT
import matplotlib.pyplot as plt
from tqdm import tqdm

from percept import YOLOPerceptionModule
from facts_converter import FactsConverter
from valuation_func import YOLOOnlineValuationFunction, YOLOClosebyValuationFunction
from logic_utils import get_index_by_predname
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression


def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default="np_pretrain",
        help="Name to store the log file as",
    )
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--ap-log", type=int, default=10, help="Number of epochs before logging AP"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-1, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["closeby_pretrain", "online_pretrain"],
        help="Use Kandinsky Pattern dataset",
    )
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
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multiple GPUs")

    parser.add_argument("--data-dir", type=str, help="Directory to data")

    parser.add_argument('--program-size', default=5, type=int,
                        help='number of clauses to compose logic programs')
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='smooth parameter in the softor function')

    args = parser.parse_args()
    return args


def compute_acc(outputs, targets):
    # print(outputs.shape)
    # print(targets.shape)
    predicts = np.argmax(outputs, axis=1)
    return accuracy_score(targets, predicts)


def preprocess(z, dataset):
    # input z: yolo_output
    # output zs: a sequece for valuation function in neural predicates
    if dataset == 'closeby_pretrain':
        return [z[:, 0], z[:, 1]]
    if dataset == 'online_pretrain':
        return [z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4]]


def run(net, predict_net, loader, optimizer, criterion, writer, args, device, train=False, epoch=0, rtpt=None, max_obj_num=4):
    iters_per_epoch = len(loader)
    loss_list = []

    be = torch.nn.BCELoss()

    loss_sum, acc_sum = 0, 0
    loss_list = []
    predicted_list = []
    target_set_list = []
    for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)
        # reset grad
        if train:
            optimizer.zero_grad()

        # yolo net to predict each object
        x = net(imgs)
        zs = preprocess(x, args.dataset)
        predicted = predict_net(*zs)
        predicted_list.append(predicted.detach().cpu().numpy())
        target_set_list.append(target_set.detach().cpu().numpy())

        # binary cross-entropy loss computation
        loss = be(predicted, target_set)
        loss_sum += loss.item()
        loss_list.append(loss.item())
        loss.backward()
        # update parameters for the step
        if optimizer != None:
            optimizer.step()

    predicted = np.concatenate(predicted_list)
    binary_output = np.where(predicted > 0.5, 1, 0)
    target_set = np.concatenate(target_set_list)

    mean_acc = accuracy_score(binary_output, target_set)
    mean_loss = loss_sum / len(loader)

    return mean_loss, 0, mean_acc


def get_data_loader(args):
    from data_kandinsky import KANDINSKY
    dataset_train = KANDINSKY(
        args.dataset, 'train'
    )
    dataset_val = KANDINSKY(
        args.dataset, 'val'
    )
    dataset_test = KANDINSKY(
        args.dataset, 'test'
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


def main():

    args = get_args()
    if args.no_cuda:
        device = 'cpu'
    else:
        device = 'cuda:' + args.device
    print('device: ', device)
    run_name = args.name + '_γ=' + str(args.gamma)
    writer = SummaryWriter(f"runs/np_pretrain/{run_name}", purge_step=0)
    # writer = None
    # utils.save_args(args, writer)

    # get torch data loader
    train_loader, val_loader,  test_loader = get_data_loader(args)

    if args.dataset == 'closeby_pretrain':
        yolo_net = YOLOPerceptionModule(e=2, d=11, device=device)
        predict_net = YOLOClosebyValuationFunction(device)
    elif args.dataset == 'online_pretrain':
        yolo_net = YOLOPerceptionModule(e=5, d=11, device=device)
        predict_net = YOLOOnlineValuationFunction(device)
    start_epoch = 0

    # parameters are inside of neural predicates
    params = list(predict_net.parameters())
    print('PARAMS: ', params)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.SmoothL1Loss()

    # Create RTPT object
    rtpt = RTPT(name_initials='HS',
                experiment_name='alphaILP/TrainNeuralPreds:' + args.dataset, max_iterations=args.epochs)
    rtpt.start()

    # train loop
    for epoch in np.arange(start_epoch, args.epochs + start_epoch):
        if not args.eval_only:
            # training step
            mean_loss, std_loss, mean_acc = run(
                yolo_net, predict_net, train_loader, optimizer, criterion, writer, args, device=device, train=True, epoch=epoch, rtpt=rtpt)
            print('training loss: ', np.round(mean_loss, 2))
            print('training acc: ', np.round(mean_acc, 2))
            writer.add_scalar("metric/train_loss",
                              mean_loss, global_step=epoch)
            writer.add_scalar("metric/train_acc",
                              mean_acc, global_step=epoch)
            rtpt.step(subtitle=f"loss={mean_loss:2.2f}")

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar(
                "lr", cur_lr, global_step=epoch * len(train_loader))
            # scheduler.step()

            # validation step
            if epoch % 5 == 0:
                mean_loss_val, std_loss_val, mean_acc_val = run(
                    yolo_net, predict_net, val_loader, None, criterion, writer, args, device=device, train=False, epoch=epoch, rtpt=rtpt)
                writer.add_scalar("metric/val_loss",
                                  mean_loss_val, global_step=epoch)
                writer.add_scalar("metric/val_acc",
                                  mean_acc_val, global_step=epoch)

                print('validation loss: ', np.round(mean_loss_val, 2))
                print('validation acc: ', np.round(mean_acc_val, 2))
                # test step
                mean_loss_test, std_loss_test, mean_acc_test = run(
                    yolo_net, predict_net, test_loader, None, criterion, writer, args, device=device, train=False, epoch=epoch, rtpt=rtpt)
                writer.add_scalar("metric/test_loss",
                                  mean_loss_test, global_step=epoch)
                writer.add_scalar("metric/test_acc",
                                  mean_acc_test, global_step=epoch)
                print('test loss: ', np.round(mean_loss_test, 2))
                print('test acc: ', np.round(mean_acc_test, 2))

        # save mlp weights for neural preficate (valuation function)
        if epoch % 10 == 0:
            save_dir = 'output/weights/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(predict_net.state_dict(),
                       save_dir + args.dataset + '.pt')


if __name__ == "__main__":
    main()
