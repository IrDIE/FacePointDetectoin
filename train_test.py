import torch
from models import ONet
from loguru import logger
from utils.dataloaders import get_dataloader, rescale_points, normalise_points
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
from utils.visualisation import  ced_plot
import cv2
import os
from tqdm import tqdm
from torchvision import transforms as tfs


EPOCHS = 100
IMG_SIZE = 128
BATCH_SIZE = 16
LR = 1e-4
EXPERIMENT_NAME = 'classic_onet'
LOGDIR = f'./Logs/tensorboard_logs/{EXPERIMENT_NAME}/'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def logg_hyperparams(tb_logger : SummaryWriter, info : dict):
    hyperparameters = {
        'IMG_SIZE': IMG_SIZE,
        'LR': LR,
        'EXPERIMENT_NAME': EXPERIMENT_NAME,
        'EPOCHS' : EPOCHS
    }
    hyperparameters.update(info)
    hyp_str = '\n'.join(['%s =  %s\n' % (key, value) for (key, value) in hyperparameters.items()])
    tb_logger.add_text(text_string = hyp_str, tag='Hyperparameters')



def update_chpt(metric_best, metric_last, model, optimizer, save_path,  label_name = 'face_net', folder_name = EXPERIMENT_NAME):
    os.makedirs(save_path + f'{folder_name}/', exist_ok=True)
    new_best = metric_best
    ckpt_info = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(ckpt_info, save_path + f'{folder_name}/{label_name}_last.pt')

    if metric_last < metric_best:
        new_best = metric_last
        torch.save(ckpt_info, save_path + f'{folder_name}/{label_name}_best.pt')

    return new_best


def test_CED(weights_path, data, model, data_name):
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    test_loader = get_dataloader(device=device, train=False, root_paths=data, rescale=IMG_SIZE, batch_size=12)
    _, error_list = mse_normed_test(test_loader, model)
    error_list = [err.item() for err in error_list]
    save_path = f'./Logs/CED_plots/{EXPERIMENT_NAME}/'
    ced_plot([error_list], save_path, title=f"Cummulative error distribution \nfor data = {data_name}\nfor exp = {EXPERIMENT_NAME} ")
    return error_list


def test(weights_path, data, model):
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model.eval()
    test_loader = get_dataloader(device=device, train=False, root_paths=data, rescale=IMG_SIZE, batch_size=BATCH_SIZE)
    return mse_normed_test(test_loader, model)


def train(n_epochs):
    model = ONet(in_ch=3, img_size=IMG_SIZE).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    train_data = ['./data/landmarks_task/300W/train_clean_crop/', './data/landmarks_task/Menpo/train_clean_crop/']
    test_data = ['./data/landmarks_task/300W/test_clean_crop/', './data/landmarks_task/Menpo/test_clean_crop/']

    train_loader = get_dataloader(device=device, train=True, root_paths=train_data, rescale=IMG_SIZE, batch_size=BATCH_SIZE)
    test_loader = get_dataloader(device=device, train=False, root_paths=test_data, rescale=IMG_SIZE, batch_size=BATCH_SIZE)
    mse_val_per_epoch_best=100

    tb_summary = SummaryWriter(LOGDIR)
    logg_hyperparams(tb_logger=tb_summary, info={'net': 'ONet'})


    for epoch in tqdm(range(n_epochs), desc="Epoch"):
        model.train()
        loss_epoch = []
        for image, target in tqdm(train_loader, position=0, leave=True, desc="Training"):
            # train
            optimizer.zero_grad()
            output = model(image)
            target = target.reshape(output.shape)
            loss = loss_fn(output, target)
            loss.backward()
            loss_epoch.append(loss.cpu().detach().numpy())
            optimizer.step()

            # image = tfs.Resize((int(128), int(128)))(image)
            # frame = np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # while True:
            #     cv2.imshow('test', frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
        # scheduler.step()
        tb_summary.add_scalar('training loss per epoch', np.mean(loss_epoch), global_step=epoch)

        # do on-end epoch validation
        model.eval()
        mse_val_per_epoch, _ = mse_normed_test(test_loader, model)

        tb_summary.add_scalar('MSE normed per epoch', mse_val_per_epoch, global_step=epoch)
        mse_val_per_epoch_best = update_chpt(mse_val_per_epoch_best, metric_last = mse_val_per_epoch, model=model, optimizer=optimizer, save_path='./ckpt/')
        logger.info(f'mse_val_per_epoch = {mse_val_per_epoch}')


def mse_normed_test(test_loader, model):
    mse_normed = []
    for image, shapes_orig, target in tqdm(test_loader, position=0, leave=True, desc="Validation:"):
        with torch.no_grad():
            output = model(image)
            # target shape should be Batch, 2
            output = output.reshape(target.shape)
            # logger.info(f'output shape= {output.shape}\n**\n')
            # add denormalise
            output = normalise_points(size_in=IMG_SIZE, points=output, denorm=True)

        # output = output.cpu().detach().numpy()

        for i, output_size in enumerate(shapes_orig):
            output_size = output_size.cpu().detach().numpy()

            # for each orig img size
            output_scaled = rescale_points(output[i], (IMG_SIZE, IMG_SIZE), output_size[1:]) # output_size = hw
            # mse for output_sized and  target[i] + normalisation on root(h*w)
            loss_per_image = mse_loss(output_scaled, target[i])
            mse_loss_per_image = loss_per_image / (math.sqrt(int(output_size[1]) * int(output_size[2])))
            mse_loss_per_image = mse_loss_per_image.cpu().detach().numpy()
            mse_normed.append(mse_loss_per_image)


    return np.mean(mse_normed), mse_normed




if __name__=="__main__":

    train(n_epochs=EPOCHS)
    # model = ONet(in_ch=3, img_size=IMG_SIZE).to(device)
    # # test_data = ['./data/landmarks_task/300W/test_clean_crop/', './data/landmarks_task/Menpo/test_clean_crop/']
    # test_data = ['./data/landmarks_task/test_ced_plot/']
    # error_list = test_CED(weights_path='./ckpt/face_net_best.pt', model= model, data = test_data, data_name = 'test_ced_plot')
    #









