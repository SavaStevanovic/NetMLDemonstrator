import numpy as np
import torch
import os
from data_loader.instance_eval import evaluate
from model_fitting.losses import SegmentationLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.images_display import join_images, visualize_label
from model_fitting.configuration import TrainingConfiguration
from torchsummary import summary
import matplotlib.pyplot as plt
from utils import plt_to_np
from model_fitting import metrics
import seaborn as sn
import torchmetrics


def process_output(lab):
    lab = lab > 0.5

    return lab.flatten()


def output_to_image(lab):
    lab = (lab > 0.5).int()
    color_map = [[0, 0, 0], [0, 0.99, 0]]
    lab = visualize_label(color_map, lab.numpy())

    return lab


def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    metric = torchmetrics.Accuracy(task ="binary", num_labels=len(net.labels), average=None).cuda()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    accs = []
    slices = [slice(0,2), slice(2,4), slice(4,7), slice(7,10), slice(10,13)]

    for i, data in enumerate(tqdm(dataloader)):
        images, labels = data
        # print(image.shape)
        images = torch.stack(images)
        image_id = [x.pop("patient_id") for x in labels]
        labels = torch.stack([torch.tensor(list(x.values())) for x in labels]).float().cuda()
        if train:
            optimizer.zero_grad()
        cuda_image = images.cuda().unsqueeze(1)
        output = net(cuda_image)
        if train:
            loss = sum(criterion(output[:, s], labels[:, s]) for s in slices)
            loss.backward()
            losses += loss.item()
            optimizer.step()
        processed_out = torch.zeros_like(output)
        for s in slices:
            one_id = output[:, s].argmax(0)
            processed_out[:, s][one_id[0], one_id[1]] = 1
        accs.extend(list((processed_out==labels).float().mean(1)))

    data_len = len(dataloader)
    run_metrics = {
        "loss": losses / data_len,
        "accs": (sum(accs)/len(accs)).item(),
    }
    return run_metrics


def fit(
    net, trainloader, validationloader, split, epochs=1000, lower_learning_period=10
):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join("checkpoints", model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, f"checkpoints_{split}.pth")
    checkpoint_conf_path = os.path.join(chp_dir, f"configuration_{split}.json")
    train_config = TrainingConfiguration()
    if os.path.exists(checkpoint_name_path):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    # summary(net, (3, 512, 512))
    writer = SummaryWriter(os.path.join("logs", model_dir_header))
    # images, _ = next(iter(trainloader))

    # with torch.no_grad():
    #     writer.add_graph(net, images[0][:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        metrics = fit_epoch(
            net, trainloader, train_config.learning_rate, train=True, epoch=epoch
        )
        writer.add_scalars(
            f"Train/Metrics/{split}",
            metrics,
            epoch,
        )
        metrics = fit_epoch(
            net, validationloader, None, train=False, epoch=epoch
        )
        writer.add_scalars(
            f"Validation/Metrics/{split}",
            metrics,
            epoch,
        )
        chosen_metric = "accs"
        os.makedirs((chp_dir), exist_ok=True)
        if (train_config.best_metric is None) or (
            train_config.best_metric < metrics[chosen_metric]
        ):
            train_config.iteration_age = 0
            train_config.best_metric = metrics[chosen_metric]
            print(
                "Epoch {}. Saving model with metric: {}".format(
                    epoch, metrics[chosen_metric]
                )
            )
            torch.save(net, checkpoint_name_path.replace(".pth", "_final.pth"))
        else:
            train_config.iteration_age += 1
            print("Epoch {} metric: {}".format(epoch, metrics[chosen_metric]))
        # net.unlock_layer()
        # net.unlock_layer()
        # if train_config.iteration_age > lower_learning_period:
        #     net.unlock_layer()
        #     net.unlock_layer()
        #     net.unlock_layer()
        #     net.unlock_layer()
        #     net.unlock_layer()
        if (train_config.iteration_age != 0) and (
            (train_config.iteration_age % lower_learning_period) == 0
        ):
            train_config.learning_rate *= 0.5
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        if train_config.iteration_age == train_config.stop_age:
            print("Stoping training")
            return
        train_config.epoch = epoch + 1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(
            net.state_dict(),
            checkpoint_name_path.replace(".pth", "_final_state_dict.pth"),
        )
    print("Finished Training")
