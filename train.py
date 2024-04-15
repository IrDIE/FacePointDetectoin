import torch
from models import ONet
from loguru import logger
from utils.dataloaders import get_dataloader, rescale_points, normalise_points
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import cv2
import os
from tqdm import tqdm
from torchvision import transforms as tfs


EPOCHS = 10
IMG_SIZE = 128
BATCH_SIZE = 2
LR = 1e-4
LOGDIR = './tensorboard_logs/'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train_step():
    pass

def train_one_epoch():
    pass


def update_chpt(metric_best, metric_last, model, optimizer, save_path, label_name = 'face_net'):
    os.makedirs(save_path, exist_ok=True)
    new_best = metric_best
    ckpt_info = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(ckpt_info, save_path + f'{label_name}_last.pt')

    if metric_last < metric_best:
        new_best = metric_last
        torch.save(ckpt_info, save_path + f'{label_name}_best.pt')

    return new_best


def test(weights_path):
    model = ONet(in_ch=3, img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    test_data = ['./data/landmarks_task/300W/test_crop_crop/']
    test_loader = get_dataloader(device=device, train=False, root_paths=test_data, rescale=IMG_SIZE, batch_size=2)
    for image, shapes_orig, target in test_loader:
        with torch.no_grad():
            output = model(image)
            # target shape should be Batch, 2
            output = output.reshape(target.shape)
            # logger.info(f'output shape= {output.shape}\n**\n')
            # add denormalise
            output = normalise_points(size_in=IMG_SIZE, points=output, denorm=True)

        for i, output_size in enumerate(shapes_orig):
            output_size = output_size.cpu().detach().numpy()
            # for each orig img size
            output_scaled = rescale_points(output[i], (IMG_SIZE, IMG_SIZE), output_size[1:]) # output_size = hw

            image = tfs.Resize((int(output_size[1]), int(output_size[2])))(image)
            frame = np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for i, point in enumerate(output_scaled):
                print(i)
                point = list(map(int, point.cpu().detach().numpy()))
                color = (227, 68, 100)
                if i == 30:
                    color = (0, 255, 0)
                frame = cv2.circle(frame, point, radius=2, color=color, thickness=-1)
            while True:
                cv2.imshow('test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def train(n_epochs, logger):
    model = ONet(in_ch=3, img_size=IMG_SIZE).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_data = ['./data/landmarks_task/300W/test_crop_crop/']
    test_data = ['./data/landmarks_task/300W/test_crop_crop/']



    train_loader = get_dataloader(device=device, train=True, root_paths=train_data, rescale=IMG_SIZE, batch_size=2)
    test_loader = get_dataloader(device=device, train=False, root_paths=test_data, rescale=IMG_SIZE, batch_size=2)
    mse_val_per_epoch_best=100

    for epoch in tqdm(range(n_epochs)):
        model.train()
        for image, target in train_loader:
            # train
            optimizer.zero_grad()
            output = model(image)
            target = target.reshape(output.shape)
            loss = loss_fn(output, target)
            tb_summary.add_scalar('training loss',loss, global_step=epoch)
            loss.backward()
            optimizer.step()

            # image = tfs.Resize((int(128), int(128)))(image)
            # frame = np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # while True:
            #     cv2.imshow('test', frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
        # scheduler.step()

        # do on-end epoch validation
        model.eval()
        mse_val_per_epoch = mse_normed_test(test_loader, model)
        mse_val_per_epoch_best = update_chpt(mse_val_per_epoch_best, metric_last = mse_val_per_epoch, model=model, optimizer=optimizer, save_path='./ckpt/')
        tb_summary.add_scalar('MSE normed per epoch',mse_val_per_epoch, global_step=epoch)


def mse_normed_test(test_loader, model):
    mse_normed = []
    for image, shapes_orig, target in test_loader:
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


    return np.mean(mse_normed)




if __name__=="__main__":
    tb_summary = SummaryWriter(LOGDIR)
    train(n_epochs=250, logger=tb_summary)
    # test('./ckpt/face_net_best.pt')


