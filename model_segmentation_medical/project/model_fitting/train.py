from matplotlib.patches import Rectangle
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
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    output_images = []
    label_images = []
    accs = 0
    f1_score = 0
    iou_total = 0

    for i, data in enumerate(tqdm(dataloader)):
        images, labels = data
        # print(image.shape)
        labels = [{k: v.cuda() for k, v in t.items()} for t in labels]
        if train:
            optimizer.zero_grad()
        cuda_image = [image.cuda() / 255 for image in images]
        if train:
            loss_dict = net(cuda_image, labels)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            losses += loss.item()
            optimizer.step()
        else:
            for idx in range(len(cuda_image)):
                c_image = [cuda_image[idx]]
                labs = [labels[idx]]
                outputs = net.eval()(torch.stack(c_image))
                for output in outputs:
                    selector = output["scores"] > 0.5
                    output["masks"] = output["masks"][selector]
                    output["labels"] = output["labels"][selector]
                    output["boxes"] = output["boxes"][selector]
                    output["scores"] = output["scores"][selector]
                    if not len(output["masks"]):
                        output["masks"] = torch.zeros(
                            [1] + list(output["masks"].shape)[1:]
                        ).cuda()
                outputs = torch.stack(
                    [
                        mask
                        for m in outputs
                        for _, mask in enumerate(m["masks"])
                        # if m["scores"][j] > 0.5
                    ]
                ).max(0)[0]
                labs = torch.stack([m["masks"].max(0)[0] for m in labs])
                iou_total += (
                    torchmetrics.JaccardIndex(
                        task="binary",
                        num_classes=2,
                        average="macro",
                        threshold=0.6,
                    )
                    .cuda()(outputs, labs)
                    .item()
                ) / len(cuda_image)
        if (not train) and (i >= len(dataloader) - 10):
            image = images[0].permute(1, 2, 0).detach().cpu().numpy()
            lab = output_to_image(labels[0]["masks"].max(0)[0].detach().cpu())
            out = net(cuda_image[:1])
            out = (out[0]["scores"] > 0.5).view(len(out[0]["masks"]), 1, 1, 1) * (
                out[0]["masks"]
            )
            out = output_to_image(out.max(0)[0][0].detach().cpu())

            _, ax = plt.subplots()
            ax.imshow(image)
            ax.imshow(lab, alpha=0.55)

            for row in labels[0]["boxes"]:
                x_min, y_min, x_max, y_max = row.detach().cpu().numpy()
                width = x_max - x_min
                height = y_max - y_min
                rect = Rectangle(
                    (x_min, y_min),
                    width,
                    height,
                    edgecolor="r",
                    facecolor="none",
                    linewidth=3,
                )
                ax.add_patch(rect)

            label_images.append(plt_to_np(plt))

            _, ax = plt.subplots()
            ax.imshow(image)
            ax.imshow(out, alpha=0.55)

            for row in labels[0]["boxes"]:
                x_min, y_min, x_max, y_max = row.detach().cpu().numpy()
                width = x_max - x_min
                height = y_max - y_min
                rect = Rectangle(
                    (x_min, y_min),
                    width,
                    height,
                    edgecolor="r",
                    facecolor="none",
                    linewidth=3,
                )
                ax.add_patch(rect)

            output_images.append(plt_to_np(plt))

    data_len = len(dataloader)
    run_metrics = {
        "loss": losses / data_len,
        "iou": iou_total / data_len,
        "accs": accs,
        "f1_score": f1_score,
    }
    return (run_metrics, output_images, label_images)


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
    images, _ = next(iter(trainloader))

    # with torch.no_grad():
    #     writer.add_graph(net, images[0][:2].cuda())
    for epoch in range(train_config.epoch, epochs):
        (metrics, output_images, label_images) = fit_epoch(
            net, trainloader, train_config.learning_rate, train=True, epoch=epoch
        )
        writer.add_scalars(
            f"Train/Metrics/{split}",
            metrics,
            epoch,
        )
        (metrics, output_images, label_images) = fit_epoch(
            net, trainloader, None, train=False, epoch=epoch
        )
        writer.add_scalars(
            f"Train_infer/Metrics/{split}",
            metrics,
            epoch,
        )
        grid = join_images(label_images)
        writer.add_images(f"train_labels/{split}", grid, epoch, dataformats="HWC")
        grid = join_images(output_images)
        writer.add_images(f"train_outputs/{split}", grid, epoch, dataformats="HWC")

        (metrics, output_images, label_images) = fit_epoch(
            net, validationloader, None, train=False, epoch=epoch
        )
        writer.add_scalars(
            f"Validation/Metrics/{split}",
            metrics,
            epoch,
        )
        grid = join_images(label_images)
        writer.add_images(f"validation_labels/{split}", grid, epoch, dataformats="HWC")
        grid = join_images(output_images)
        writer.add_images(f"validation_outputs/{split}", grid, epoch, dataformats="HWC")
        chosen_metric = "iou"
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
