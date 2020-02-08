import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def predict_res(bi_gru, data_loader, device):
    bi_gru.eval()
    y_true = None
    y_pred = None
    with torch.no_grad():
        for batch in data_loader:
            seqs, seq_lens, tgts = batch
            seqs = seqs.to(device)
            tgts = tgts.to(device)
            if y_true is None:
                y_true = tgts
            else:
                y_true = torch.cat((y_true, tgts), 0)
            outputs = bi_gru(seqs, seq_lens)
            if y_pred is None:
                y_pred = outputs
            else:
                y_pred = torch.cat((y_pred, outputs), 0)

    print(y_true.size(), y_pred.size())
    y_pred = torch.sigmoid(y_pred)
    return y_true.cpu().numpy(), y_pred.cpu().numpy()


def mean_column_wise_auc(y_true, y_pred):
    assert y_true.shape[1] == y_pred.shape[1], 'Arrays must have the same dimension'
    list_of_aucs = []
    for column in range(y_true.shape[1]):
        #print(sum(y_true[:,column]), sum(y_pred[:,column]))
        if sum(y_true[:,column]) == 0:
            continue
        list_of_aucs.append(roc_auc_score(y_true[:,column],y_pred[:,column]))
    # print(list_of_aucs)
    return np.array(list_of_aucs).mean(), len((list_of_aucs))


def fit_active_value(s1, active_value=0.02):
    # return np.round(s1)
    return (s1 > active_value).astype(int)


def cut_max_6(s1):
    # get first 6 max probability
    max6ids = np.argsort(s1)[-6:]
    s3 = np.zeros_like(s1)
    # for i in max6ids:
    #     s3[i] = 1
    s3[max6ids] = 1
    return s3


def cal_avg_p_r(arr_true, arr_pred):
    ps, rs = [], []
    for i in range(arr_true.shape[0]):
        t1, s1 = arr_true[i], arr_pred[i]
        if sum(t1) <= 0:
            continue
        s2 = fit_active_value(s1)
        # if sum(s2) > 6: # multi-labels > 6
        #     s2 = cut_max_6(s1)
        p, r = precision_score(t1, s2), recall_score(t1, s2)
        ps.append(p)
        rs.append(r)
    return np.average(ps), np.average(rs), len(ps), ps, rs

# ........ some code ignored ...............
PATH = save_path
bi_gru.load_state_dict(torch.load(PATH))

val_true, val_pred = predict_res(bi_gru, val_loader, device)

# cal roc_auc_score if every label include {0, 1}
auc = roc_auc_score(val_true, val_pred)

# if some label just {0}
auc = mean_column_wise_auc(val_true, val_pred)

# cal average precision and recall
avg_p, avr_r, num, ps, rs = cal_avg_p_r(val_true, val_pred)
