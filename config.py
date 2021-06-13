import os
import inspect


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

learning_rate = 5e-4

data_path = os.path.join(current_dir, 'MNIST_data')

y_label_list = [str(i) for i in range(10)]

train_batch_size = 16

val_batch_size = 256

cpu_workers = 1

reload_checkpoint = ''

total_epochs = 5

valid_interval = 500

show_interval = 10

save_interval = 2000

width = 28

height = 28

input_shape = (-1, 3, width, height)

save_model_path = 'model_weights'

reload_checkpoint = os.path.join(current_dir, save_model_path, 'cnn_012001_loss0.2541285455226898.pth')
