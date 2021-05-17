"""
training process for CNN
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import config as cfg
from data_manager import ClassficationDataset, mnist_collate
from model import CNN
from evaluate import evaluate
from utils import check_path_exists


def save_model_weight(model_path, model_name, model, iteration, loss):
    """save input model weight"""
    save_model_path = os.path.join(
        model_path,
        f'{model_name}_{iteration:06}_loss{loss}.pth'
    )
    torch.save(model.state_dict(), save_model_path)
    print(f'save model at {save_model_path}')


def train_batch(cnn, data, optimizer, criterion, device, input_shape):
    """
    input_shape = (-1, 3, 28, 28)
    """
    cnn.train()
    images, y_labels = [d.to(device) for d in data]
    images = Variable(images).view(input_shape)
    y_labels = Variable(y_labels)

    logits = cnn(images)

    loss = criterion(logits, y_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def prepare_training_data(cfg):
    """return train & val datasets"""
    mnist_train_dataset = ClassficationDataset(cfg.data_path, 'train', cfg.y_label_list)
    mnist_val_dataset = ClassficationDataset(cfg.data_path, 'val', cfg.y_label_list)

    train_loader = DataLoader(
        dataset=mnist_train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        # num_workers=cfg.cpu_workers,
        collate_fn=mnist_collate
    )

    val_loader = DataLoader(
        dataset=mnist_val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=True,
        # num_workers=cfg.cpu_workers,
        collate_fn=mnist_collate
    )
    return train_loader, val_loader


def main(cfg):
    """
    main training process for CNN
    """
    cnn = CNN()
    train_loader, val_loader = prepare_training_data(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reload_checkpoint = None
    if reload_checkpoint:
        cnn.load_state_dict(torch.load(cfg.reload_checkpoint, map_location=device))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    check_path_exists(cfg.save_model_path)
    i = 1
    for epoch in range(1, cfg.total_epochs + 1):
        print(f'Training epoch: {epoch}')
        total_train_loss, total_train_count = 0, 0
        for train_data in train_loader:
            loss = train_batch(
                cnn, train_data, optimizer, criterion, device, cfg.input_shape
            )
            train_size = train_data[0].size(0)
            total_train_loss += loss
            total_train_count += train_size

            if i % cfg.show_interval == 0:
                print(f'train_batch_loss[{i}]: {loss / train_size}')

            if i % cfg.valid_interval == 0:
                evaluation = evaluate(
                    cnn, val_loader, criterion, cfg.input_shape, max_iteration=10
                )
                print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

                if i % cfg.save_interval == 0:
                    save_model_weight(
                        model_path=cfg.save_model_path,
                        model_name='cnn',
                        model=cnn,
                        iteration=i,
                        loss=loss
                    )
            i += 1

        save_model_weight(
            model_path=cfg.save_model_path,
            model_name='cnn',
            model=cnn,
            iteration=i,
            loss=loss
        )

        print('train_loss: ', total_train_loss / total_train_count)


if __name__ == '__main__':
    import time
    t0 = time.time()
    main(cfg)
    t1 = time.time()
    print('It costs {} minutes'.format((t1-t0)/60))
