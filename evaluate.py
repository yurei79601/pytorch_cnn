"""
functions for evaluate model performance when training
"""
from tqdm import tqdm
import torch
from torch.autograd import Variable


def evaluate(cnn, dataloader, criterion, input_shape, max_iteration=None):
    """
    measure loss and accuracy for the input model
    """
    cnn.eval()

    total_val_correct_cnt = 0
    total_test_cnt = 0
    total_val_loss = 0
    total_check_iteration = max_iteration if max_iteration else len(dataloader)
    pbar = tqdm(total=total_check_iteration, desc="Evaluate")

    with torch.no_grad(): 
        for i, (images, y_labels) in enumerate(dataloader):
            if i >= total_check_iteration:
                break
            # 1.Define variables
            images = Variable(images).view(input_shape)
            y_labels = Variable(y_labels)
            # 2.Forward propagation
            logits = cnn(images)
            # 3.Calculate softmax and cross entropy loss
            total_val_loss += criterion(logits, y_labels)
            # 4.Get predictions from the maximum value
            predict_labels = torch.argmax(logits, dim=1)
            # 5.Total number of labels
            total_test_cnt += len(y_labels)
            total_val_correct_cnt += int(torch.sum(y_labels == predict_labels))
            pbar.update(1)
        pbar.close()
    evaluation = {
        'loss': total_val_loss / total_test_cnt,
        'acc': total_val_correct_cnt / total_test_cnt,
        #'wrong_cases': wrong_cases
    }
    return evaluation
