"""
functions for CNN prediction
main function: main
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import config as cfg
from model import CNN


def load_cnn_weight(reload_checkpoint, device=torch.device('cpu')):
    cnn = CNN()
    cnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    return cnn


def cnn_preprocessing(images: torch.tensor, input_shape: tuple):
    assert images.dtype.is_floating_point, 'The input images must of type torch.float'
    return images.reshape(cfg.input_shape)


def cnn_decoder(logits_probs: torch.Tensor):
    return [cfg.y_label_list[i] for i in torch.argmax(logits_probs, dim=1)]


def main(cnn, images: torch.Tensor):
    """To get CNN predict result"""
    images = cnn_preprocessing(torch.FloatTensor(images), cfg.input_shape)
    logits = cnn(images)
    logits_probs = F.log_softmax(logits, dim=1)
    return cnn_decoder(logits_probs)


if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader
    from data_manager import ClassficationDataset, mnist_collate
    mnist_val_dataset = ClassficationDataset(cfg.data_path, 'val', cfg.y_label_list)

    val_loader = DataLoader(
            dataset=mnist_val_dataset,
            batch_size=cfg.val_batch_size,
            shuffle=True,
            # num_workers=cfg.cpu_workers,
            collate_fn=mnist_collate
    )

    for i, (images, y_labels) in enumerate(val_loader):
        if i >= 1:
            break

    cnn = load_cnn_weight(cfg.reload_checkpoint)
    t0 = time.time()
    predicted = main(cnn, images)
    t1 = time.time()

    print(f'The true answer list is: {predicted}')
    print('It prediction preocess costs {} seconds, {} in average'.format(t1-t0, (t1-t0)/len(predicted)))
    print('complete')
