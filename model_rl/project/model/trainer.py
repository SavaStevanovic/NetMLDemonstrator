import os
from torch import nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.configuration import TrainingConfiguration


class WeightedMSELoss(nn.Module):

    def forward(self, output, target):
        mean = 0
        std = target.std(0)
        std[std == 0] = 1
        output = (output - mean)/std
        target = (target - mean)/std
        loss = (output - target) ** 2
        return loss.mean()


def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    criterion = WeightedMSELoss()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    for i, data in enumerate(tqdm(dataloader)):
        cur_state, next_states, act, reward, _ = data
        if train:
            optimizer.zero_grad()
        outputs = net(torch.cat((cur_state, act), dim=-1))
        roward_label = torch.cat(
            (
                torch.tensor(reward, requires_grad=False).unsqueeze(-1),
                (next_states - cur_state),
                
            ), dim=-1
        )
        loss = criterion(outputs, roward_label)
        if train:
            loss.backward()
            optimizer.step()
        losses += loss.item()
    data_len = len(dataloader)
    return losses/data_len


def fit(net, trainloader, validationloader, dataset_name, writer: SummaryWriter, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(
        chp_dir, '{}_checkpoints.pth'.format(dataset_name))
    checkpoint_conf_path = os.path.join(
        chp_dir, '{}_configuration.json'.format(dataset_name))
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    for epoch in range(train_config.epoch, epochs):
        if train_config.iteration_age and (train_config.iteration_age % (2*lower_learning_period+1)) == 0:
            break
        loss = fit_epoch(net, trainloader,
                         train_config.learning_rate, True, epoch=epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)

        loss = fit_epoch(net, validationloader,
                         train_config.learning_rate, False, epoch)
        writer.add_scalar('Validation/Metrics/loss', loss, epoch)

        os.makedirs((chp_dir), exist_ok=True)
        if (train_config.best_metric is None) or (train_config.best_metric > loss):
            train_config.iteration_age = 0
            train_config.best_metric = loss
            print('Epoch {}. Saving model with metric: {}'.format(epoch, loss))
            torch.save(net, checkpoint_name_path.replace('.pth', '_final.pth'))
        else:
            train_config.iteration_age += 1
            print('Epoch {} metric: {}'.format(epoch, loss))
        if train_config.iteration_age and (train_config.iteration_age % lower_learning_period) == 0:
            train_config.learning_rate *= 0.5
            print("Learning rate lowered to {}".format(
                train_config.learning_rate))
        

        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(net.state_dict(), checkpoint_name_path.replace(
            '.pth', '_final_state_dict.pth'))
    loss = fit_epoch(net, validationloader,
                     train_config.learning_rate, False, 1000)
    print(f'Final metric: {loss}')
    return loss
