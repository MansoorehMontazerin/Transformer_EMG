import os
from utils_transformer import init_dataset
import torch
from Layers_transformer import Transformer
import argparse
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import copy
import scipy
from scipy import ndimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checking if the CUDA is available
torch.cuda.is_available()


# defining the required functions
def get_n_params(module):
    """
    This function returns the number of learnable parameters in the model

    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def init_model(opt):
    """
    This function defines the main model and its input parameters

     Parameters
    ----------
     opt : 
        model inputs defined in the "main" function       
   """

    model = Transformer(
        seq_len=opt.seq_len,
        max_len=opt.max_len,
        n_classes=opt.n_classes,
        embed_dim=opt.embed_dim,
        depth=opt.depth,
        grid=opt.grid,
        n_heads=opt.n_heads,
        mlp_ratio=opt.mlp_ratio,
        p=opt.p,
        attn_p=opt.attn_p,
        norm_layer=nn.LayerNorm
    )
    model = model.cuda() if opt.cuda else model
    return model


def save_list_to_file(path, thelist):
    """
    This function saves the generated lists in the program to a file on the computer

   """
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def batch_for_transformer(opt, x, y):
    """
    This function prepares batches in the dataloader to be inputted to the model.
    It also sends the input batch to CUDA.
    Permutation is done based on the shape of the input batch.
    In this specific part, no permutation is needed.

    Parameters
    ----------
    opt : 
        model inputs defined in the "main" function

    x : float32
        data of shape (n_samples,seq_len,h_grid,v_grid)

    y:  int64
        labels of shape (n_samples)
   """

    if opt.permute_img:
        x = x.permute(0, 2, 1, 3)
    if opt.cuda:
        x, y = x.cuda(), y.cuda()
    return x.float(), y.long()


def get_acc(model_output, last_targets):
    """
    This function computes the accuracy 
    Parameters
    ----------
    model_output : float32 tensor
        predictions

    last_targets : int64 tensor
        targets
    """
    _, preds = model_output.max(1)
    acc = torch.eq(preds, last_targets).float().mean()
    return acc.item()


# Training function

def train(opt, tr_dataloader, model, optim, scheduler, val_dataloader=None):
    """
    This function is for training the model.

    Parameters
    ----------
    opt : 
        model parameters defined in the "main" function

    tr_dataloader:
        training dataloader

    val_dataloader:
        validation dataloader

    optim:
        Optimizer

    Scheduler:
        Learning rate annealing scheduler object

    best_state:
        The best_model achieved in the validation part

    best_acc:
        The accuracy of the best_state

    """

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0  # The accuracy of the best model obtained by changing the hyperparameters and
    # observing the performance of the model on the validation set in each epoch.
    # This model may be achieved before reaching the last epoch and is used in early stopping

    x = []
    y = []
    # The accuracy of the best model obtained by changing the hyperparameters and
    best_model_path = os.path.join(opt.exp, 'best_model.pth')
    # observing the performance of the model on the validation set in each epoch

    # This model may be achieved before reaching the last epoch and is used in early stopping
    last_model_path = os.path.join(opt.exp, 'last_model.pth')
    # the last model is achieved in the last epoch and it may not be the best model

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(opt.epochs):
        count = 0
        print('=== Epoch: {} ==='.format(epoch+1))
        tr_iter = iter(tr_dataloader)
        model.train()
        model = model.cuda()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = batch_for_transformer(opt, x, y)
            model_output = model(x)
            loss = loss_fn(model_output, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(get_acc(model_output, y))
            count += 1

        assert count == len(tr_dataloader)

        avg_loss = np.mean(train_loss[-len(tr_dataloader):])
        avg_acc = np.mean(train_acc[-len(tr_dataloader):])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        for param_group in optim.param_groups:
            print(param_group['lr'])

        scheduler.step()

        if val_dataloader is None:
            continue

        # Validation part
        val_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                x, y = batch
                x, y = batch_for_transformer(opt, x, y)
                model_output = model(x)
                loss = loss_fn(model_output, y)
                val_loss.append(loss.item())
                val_acc.append(get_acc(model_output, y))
        avg_loss = np.mean(val_loss[-len(val_dataloader):])
        avg_acc = np.mean(val_acc[-len(val_dataloader):])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(
                opt.exp, name + '.txt'), locals()[name])

    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


# Testing function
def test(opt, test_dataloader, model, flag):
    """
    This function is for testing the model
    Parameters
    ----------
    opt : 
        model parameters defined in the "main" function

    test_dataloader:
        testing dataloader

    flag:
      last or best
      used for saving the test accuracy. determines if the testing is done with the best or the last model

    """

    test_acc_batch = list()
    test_iter = iter(test_dataloader)
    model.eval()
    with torch.no_grad():
        for batch in test_iter:
            x, y = batch
            x, y = batch_for_transformer(opt, x, y)
            model_output = model(x)
            test_acc_batch.append(get_acc(model_output, y))

    assert len(test_dataloader) == len(test_acc_batch)
    test_acc = np.mean(test_acc_batch)
    test_std = np.std(test_acc_batch)
    for name in ['test_acc_batch']:
        save_list_to_file(os.path.join(
            opt.exp, name + '_' + flag + '.txt'), locals()[name])

    with open(os.path.join(opt.exp, 'average_test_acc' + '.txt'), 'w') as f:
        f.write("%s\n" % test_acc)
    with open(os.path.join(opt.exp, 'average_test_std' + '.txt'), 'w') as f:
        f.write("%s\n" % test_std)
    print('****************The model is***************** {}********************'.format(opt.exp))
    print('Test Acc: {}, Test Std: {}'.format(test_acc, test_std))
    print('len(test_acc_batch): {}'.format(len(test_acc_batch)))

    return test_acc, test_std


# The main function
def main(i):
    '''
    Initialize everything and train

    Parameters
    ----------
    exp : 
        The path in which the model and its attricutes are saved

    sub_idx:
        The subject index

    seq_len:
        The window_size of the data

    permute_img:
        Determines if the data in the batch should be permuted
        False for Transformer 
        True  for Vision Transformer

    max_len:
        Maximum length of the sequence (window_size) used for positional embedding

    grid:
        The size of the electrode grid

    n_classes:
        Number of gestures

    embed_dim:
        Embedding dimension of the model which is considered as the number of electrodes in the transformer part

    depth:
        Model's depth---The number of transforemr encoders that are serially concatenated to each other

    n_heads:
        Number of heads in the attention part

    mlp_ratio:
        The ratio of hidden_size/input_size in the MLP of the encoder

    p:
       Dropout probability used in the Attention and MLP blocks

    attn_p:
        Dropout probability used in the Attention block after multiplying query and key

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp', type=str, default='EMG_training_trans/model_{}_rep2/Subj_{}'.format(1, i))
    parser.add_argument('--sub_idx', type=int, default=i)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--permute_img', action='store_true', default=False)
    parser.add_argument('--max_len', type=int, default=150)
    parser.add_argument('--grid', type=int, default=(8, 8))
    parser.add_argument('--n_classes', type=int, default=66)
    parser.add_argument('--embed_dim', type=int, default=8*8)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=3)
    parser.add_argument('--p', type=float, default=0)
    parser.add_argument('--attn_p', type=float, default=0.27)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda', action='store_true',  default=True)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument(
        "-f", "--fff", help="a dummy argument to fool ipython", default="1")
    options = parser.parse_args()

    if not os.path.exists(os.path.join(options.exp)):
        os.makedirs(os.path.join(options.exp))

    with open(os.path.join(options.exp, 'options' + '.txt'), 'w') as f:
        f.write("%s\n" % options)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --CUDA")

    model = init_model(options)
    print('The number of parameters: {}'.format(get_n_params(model)))

    tr_dataloader, ts_dataloader = init_dataset(options)
    # tr_dataloader, val_dataloader= init_dataset(options)

    # if torch.cuda.device_count() > 1:
    #model = nn.DataParallel(model, device_ids=[0, 1], dim=0)

    # optim = torch.optim.Adam(params=model.parameters(
    # ), lr=options.lr, weight_decay=options.weight_decay)
    # scheduler = StepLR(optim, step_size=10, gamma=1)

    # res = train(opt=options,
    #             tr_dataloader=tr_dataloader,
    #             val_dataloader=None,
    #             model=model,
    #             optim=optim,
    #             scheduler=scheduler)
    # best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res

    model.load_state_dict(torch.load(os.path.join(options.exp, 'last_model.pth')))
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=ts_dataloader,
         model=model,
         flag='last')

    #model.load_state_dict(torch.load(os.path.join(options.exp, 'best_model.pth')))
    #print('Testing with best model..')
    # test(opt=options,
    # test_dataloader=test_dataloader,
    # model=model,
    # flag='best')

    print('The number of parameters: {}'.format(get_n_params(model)))
    print("---------------subject_idx is {}".format(options.sub_idx))
    print(options)


if __name__ == '__main__':

    for i in [1]:

        main(i)
