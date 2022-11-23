def cal_f1(num_truth, num_pred, num_tp):
    """calculate precision/recall/f1 score

    return:
        precision, recall, f1
    """
    
    if num_tp < 0 or num_pred < 0 or num_truth < 0 or num_tp > num_pred or num_tp > num_truth:
        raise ValueError('invalid parameters')
    return 100 * num_tp / (num_pred + 1e-8), 100 * num_tp / (num_truth + 1e-8), 200 * num_tp / (num_truth + num_pred + 1e-8)
    