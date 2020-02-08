import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import BiGRU
from dataset import TCdata

def train_epoch(model, device, epoch, train_loader, test_loader, criterion, optimizer, clip=5.):
    model.train()
    train_loss = 0
    t0 = time.time()
    for i, batch in enumerate(train_loader, 1):
        seqs, seq_lens, tgts = batch
        seqs = seqs.to(device)
        tgts = tgts.to(device)
        
        optimizer.zero_grad()
        outputs = model(seqs, seq_lens)
        loss = criterion(outputs, tgts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        train_loss += loss.item()
        
        if i % 100 == 0:
            # print loss info every 20 Iterations
            log_str = "Epoch : {} , Iteration : {} , Time : {:.2f} , TrainLoss : {:.4f}".format \
                        (epoch, i, time.time()-t0, train_loss/i)
            print(log_str)
            t0 = time.time()
    train_loss /= len(train_loader)
    
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            seqs, seq_lens, tgts = batch
            seqs = seqs.to(device)
            tgts = tgts.to(device)

            outputs = model(seqs, seq_lens)
            loss = criterion(outputs, tgts)
            eval_loss += loss.item()
            
    eval_loss /= len(val_loader)
    return model, optimizer, train_loss, eval_loss


def load_data(data_path, batch_sz=512):
    with open(data_path, 'rb') as f:
        process_data = pickle.load(f)

    x_train, x_val, y_train, y_val = process_data['x_train'], process_data['x_val'], process_data['y_train'], process_data['y_val']
    # word2id, id2word = process_data['word2id'], process_data['id2word']

    emb_weights = np.load('EmbeddingMatrix.npy')
    return x_train, x_val, y_train, y_val, emb_weights


def train():
    x_train, x_val, y_train, y_val, emb_weights = load_data('data_path')

    train_loader = DataLoader(TCdata(x_train, y_train), 
                              num_workers=1, 
                              batch_size=batch_sz, 
                              collate_fn=pair_collate_func, 
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(TCdata(x_val, y_val), 
                            num_workers=1, 
                            batch_size=10, # match len(x_val)
                            collate_fn=pair_collate_func, 
                            shuffle=True,
                            drop_last=True)

    emb_dim = 300
    hidden_size = 64
    n_layers = 1
    num_classes = 6
    dropout = 0.5
    bi_gru = BiGRU(emb_dim, hidden_size, n_layers, num_classes, dropout, emb_weights)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    bi_gru.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(bi_gru.parameters())

    n_epochs = 15
    best_eval_loss = float('inf')
    for epoch in range(1, 1+n_epochs):
        bi_gru, optimizer, train_loss, eval_loss = train_epoch(bi_gru, device, epoch, train_loader, val_loader, 
                                                            criterion, optimizer, clip=5.)

        print(">> Epoch : {} , TrainLoss : {:.4f} , EvalLoss : {:.4f}\n".format \
            (epoch, train_loss, eval_loss))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(bi_gru.state_dict(), 'bi_gru_0206.pt')


if __name__ == '__main__':
    train()