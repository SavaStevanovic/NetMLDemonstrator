import torch
import os
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


def process_output(lab, flat_mask):
    lab = lab > 0.5

    return lab.flatten()[flat_mask]


def output_to_image(lab):
    lab = (lab > 0.5).int().squeeze(0)
    color_map = [[0, 0, 0], [0, 0.99, 0]]
    lab = visualize_label(color_map, lab.numpy())

    return lab


def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    cm = metrics.RunningConfusionMatrix()
    criterion = SegmentationLoss()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    output_images = []
    label_images = []
    accs = 0
    f1_score = 0

    for i, data in enumerate(tqdm(dataloader)):
        image, labels, dataset_id = data
        # print(image.shape)
        if train:
            optimizer.zero_grad()
        mask = 1 - labels[:, -1]
        labels_cuda = labels.cuda()
        labels_cuda = labels_cuda[:, 1:]
        mask_cuda = mask.cuda()
        labels_cuda = labels_cuda[:, :-1]
        outputs = net(image.cuda())
        outputs = outputs * mask_cuda.unsqueeze(1)
        loss, focal_loss, dice_loss = criterion(outputs, labels_cuda)
        if train:
            loss.backward()
            optimizer.step()
        losses += loss.item()
        total_focal_loss += focal_loss
        total_dice_loss += dice_loss

        flat_mask = mask_cuda.flatten().bool()
        lab = process_output(labels_cuda, flat_mask)
        out = process_output(outputs.sigmoid(), flat_mask)

        cm.update_matrix(lab.int(), out.int())

        if i >= len(dataloader) - 10:
            image = image[0].permute(1, 2, 0).detach().cpu().numpy()
            lab = output_to_image(labels[0, 1:-1].detach().cpu())
            out = output_to_image(outputs[0].detach().sigmoid().cpu())

            plt.imshow(image)
            plt.imshow(lab, alpha=0.55)
            label_images.append(plt_to_np(plt))

            plt.imshow(image)
            plt.imshow(out, alpha=0.55)
            output_images.append(plt_to_np(plt))

    f1_score, accs, confusion_matrix = cm.compute_metrics()
    sn.heatmap(confusion_matrix, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_img = plt_to_np(plt)

    data_len = len(dataloader)
    run_metrics = {
        "loss": losses / data_len,
        "focal_loss": total_focal_loss / data_len,
        "dice_loss": total_dice_loss / data_len,
        "accs": accs,
        "f1_score": f1_score,
    }
    return (
        run_metrics,
        output_images,
        label_images,
        cm_img,
    )


def fit(net, trainloader, validationloader, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join("checkpoints", model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, "checkpoints.pth")
    checkpoint_conf_path = os.path.join(chp_dir, "configuration.json")
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    summary(net, (3, 512, 512))
    writer = SummaryWriter(os.path.join("logs", model_dir_header))
    images, _, _ = next(iter(trainloader))

    with torch.no_grad():
        writer.add_graph(net, images[:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        (
            metrics,
            output_images,
            label_images,
            cm,
        ) = fit_epoch(
            net, trainloader, train_config.learning_rate, train=True, epoch=epoch
        )
        writer.add_scalars(
            "Train/Metrics",
            metrics,
            epoch,
        )
        grid = join_images(label_images)
        writer.add_images("train_labels", grid, epoch, dataformats="HWC")
        grid = join_images(output_images)
        writer.add_images("train_outputs", grid, epoch, dataformats="HWC")
        writer.add_images("train_confusion_matrix", cm, epoch, dataformats="HWC")
        (
            metrics,
            output_images,
            label_images,
            cm,
        ) = fit_epoch(net, validationloader, None, train=False, epoch=epoch)
        writer.add_scalars(
            "Validation/Metrics",
            metrics,
            epoch,
        )
        grid = join_images(label_images)
        writer.add_images("validation_labels", grid, epoch, dataformats="HWC")
        grid = join_images(output_images)
        writer.add_images("validation_outputs", grid, epoch, dataformats="HWC")
        writer.add_images("validation_confusion_matrix", cm, epoch, dataformats="HWC")

        os.makedirs((chp_dir), exist_ok=True)
        if (train_config.best_metric is None) or (
            train_config.best_metric < metrics["f1_score"]
        ):
            train_config.iteration_age = 0
            train_config.best_metric = metrics["f1_score"]
            print(
                "Epoch {}. Saving model with metric: {}".format(
                    epoch, metrics["f1_score"]
                )
            )
            torch.save(net, checkpoint_name_path.replace(".pth", "_final.pth"))
        else:
            train_config.iteration_age += 1
            print("Epoch {} metric: {}".format(epoch, metrics["f1_score"]))
        if train_config.iteration_age == lower_learning_period:
            train_config.learning_rate *= 0.5
            train_config.iteration_age = 0
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        train_config.epoch = epoch + 1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(
            net.state_dict(),
            checkpoint_name_path.replace(".pth", "_final_state_dict.pth"),
        )
    print("Finished Training")
