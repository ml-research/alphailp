import argparse

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_curve

import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rtpt import RTPT

from nsfr_utils import denormalize_kandinsky, get_data_loader, get_data_pos_loader, get_prob, get_nsfr_model, update_initial_clauses
from nsfr_utils import save_images_with_captions, to_plot_images_kandinsky, generate_captions
from logic_utils import get_lang, get_searched_clauses
from mode_declaration import get_mode_declarations

from clause_generator import ClauseGenerator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--e", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", choices=["twopairs", "threepairs", "red-triangle", "closeby",
                                              "online", "online-pair", "nine-circles", "clevr-hans0", "clevr-hans1", "clevr-hans2"], help="Use kandinsky patterns dataset")
    parser.add_argument("--dataset-type", default="kandinsky",
                        help="kandinsky or clevr")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--small-data", action="store_true",
                        help="Use small training data.")
    parser.add_argument("--no-xil", action="store_true",
                        help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1,
                        help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=2,
                        help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101,
                        help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    args = parser.parse_args()
    return args

# def get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False):


def discretise_NSFR(NSFR, args, device):
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses_, bk_clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset)
    # Discretise NSFR rules
    clauses = NSFR.get_clauses()
    return get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False)


def predict(NSFR, loader, args, device,  th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0
    ###NSFR = discretise_NSFR(NSFR, args, device)
    # NSFR.print_program()

    for i, sample in tqdm(enumerate(loader, start=0)):
        # to cuda
        imgs, target_set = map(lambda x: x.to(device), sample)

        # infer and predict the target probability
        V_T = NSFR(imgs)
        predicted = get_prob(V_T, NSFR, args)
        predicted_list.append(predicted.detach())
        target_list.append(target_set.detach())
        if args.plot:
            imgs = to_plot_images_kandinsky(imgs)
            captions = generate_captions(
                V_T, NSFR.atoms, NSFR.pm.e, th=0.3)
            save_images_with_captions(
                imgs, captions, folder='result/kandinsky/' + args.dataset + '/' + split + '/', img_id_start=count, dataset=args.dataset)
        count += V_T.size(0)  # batch size

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


def train_nsfr(args, NSFR, optimizer, train_loader, val_loader, test_loader, device, writer, rtpt):
    bce = torch.nn.BCELoss()
    loss_list = []
    for epoch in range(args.epochs):
        loss_i = 0
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            # to cuda
            imgs, target_set = map(lambda x: x.to(device), sample)
            imgs.requires_grad = True
            # infer and predict the target probability
            V_T = NSFR(imgs)
            # NSFR.print_valuation_batch(V_T)
            predicted = get_prob(V_T, NSFR, args)
            loss = bce(predicted, target_set)
            loss_i += loss.item()
            loss.backward()

            print(imgs.grad.shape)
            print(imgs.grad)
            optimizer.step()

            # if i % 20 == 0:
            #    NSFR.print_valuation_batch(V_T)
            #    print("predicted: ", np.round(predicted.detach().cpu().numpy(), 2))
            #    print("target: ", target_set.detach().cpu().numpy())
            #    NSFR.print_program()
            #    print("loss: ", loss.item())

            #print("Predicting on validation data set...")
            # acc_val, rec_val, th_val = predict(
            #    NSFR, val_loader, args, device, writer, th=0.33, split='val')
            #print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
        loss_list.append(loss_i)
        rtpt.step(subtitle=f"loss={loss_i:2.2f}")
        writer.add_scalar("metric/train_loss", loss_i, global_step=epoch)
        print("loss: ", loss_i)
        # NSFR.print_program()
        if epoch % 20 == 0:
            NSFR.print_program()
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(
                NSFR, val_loader, args, device, th=0.33, split='val')
            writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            print("acc_val: ", acc_val)

            print("Predicting on training data set...")
            acc, rec, th = predict(
                NSFR, train_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            print("acc_train: ", acc)

            print("Predicting on test data set...")
            acc, rec, th = predict(
                NSFR, test_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/test_acc", acc, global_step=epoch)
            print("acc_test: ", acc)

    return loss


def main(n):
    args = get_args()
    if args.dataset_type == 'kandinsky':
        if args.small_data:
            name = 'small_KP/aILP:' + args.dataset + '_' + str(n)
        else:
            name = 'KP/aILP:' + args.dataset + '_' + str(n)
    else:
        if not args.no_xil:
            name = 'CH/aILP:' + args.dataset + '_' + str(n)
        else:
            name = 'CH/aILP-noXIL:' + args.dataset + '_' + str(n)
    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)

    print('device: ', device)
    #run_name = 'predict/' + args.dataset
    writer = SummaryWriter(f"runs/{name}", purge_step=0)

    # Create RTPT object
    rtpt = RTPT(name_initials='HS', experiment_name=name,
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # get torch data loader
    train_loader, val_loader,  test_loader = get_data_loader(args)

    train_pos_loader, val_pos_loader, test_pos_loader = get_data_pos_loader(
        args)
    #####train_pos_loader, val_pos_loader, test_pos_loader = get_data_loader(args)

    # load logical representations
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses, bk_clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset)
    clauses = update_initial_clauses(clauses, args.n_obj)
    print("clauses: ", clauses)

    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(
        args, lang, clauses, atoms, bk, bk_clauses, device=device)  # torch.device('cpu'))
    mode_declarations = get_mode_declarations(args, lang, args.n_obj)
    cgen = ClauseGenerator(args, NSFR_cgen, lang, val_pos_loader, mode_declarations,
                           bk_clauses, device=device, no_xil=args.no_xil)  # torch.device('cpu'))
    # generate clauses
    if args.pre_searched:
        clauses = get_searched_clauses(
            lark_path, lang_base_path, args.dataset_type, args.dataset)
    else:
        clauses = cgen.generate(
            clauses, T_beam=args.t_beam, N_beam=args.n_beam, N_max=args.n_max)
    print("====== ", len(clauses), " clauses are generated!! ======")
    # update
    NSFR = get_nsfr_model(args, lang, clauses, atoms, bk,
                          bk_clauses, device, train=True)

    params = NSFR.get_params()
    optimizer = torch.optim.RMSprop(params, lr=args.lr)
    ##optimizer = torch.optim.Adam(params, lr=args.lr)

    loss_list = train_nsfr(args, NSFR, optimizer, train_loader,
                           val_loader, test_loader, device, writer, rtpt)

    # validation split
    print("Predicting on validation data set...")
    acc_val, rec_val, th_val = predict(
        NSFR, val_loader, args, device, th=0.33, split='val')

    print("Predicting on training data set...")
    # training split
    acc, rec, th = predict(
        NSFR, train_loader, args, device, th=th_val, split='train')

    print("Predicting on test data set...")
    # test split
    acc_test, rec_test, th_test = predict(
        NSFR, test_loader, args, device, th=th_val, split='test')

    print("training acc: ", acc, "threashold: ", th, "recall: ", rec)
    print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
    print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)


if __name__ == "__main__":
    for i in range(5):
        main(n=i)
