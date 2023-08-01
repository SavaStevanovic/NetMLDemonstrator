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
    cm = metrics.RunningConfusionMatrix()
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
        cuda_image = [image.cuda() for image in images]
        if train:
            loss_dict = net(cuda_image, labels)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            losses += loss.item()
            optimizer.step()
        else:
            outputs = net.eval()(torch.stack(cuda_image))
            outputs = torch.stack(
                [torch.zeros([1] + list(labels[0]["masks"].shape[1:])).cuda()]
                + [
                    mask
                    for m in outputs
                    for i, mask in enumerate(m["masks"])
                    if m["scores"][i] > 0.5
                ]
            ).max(0)[0]
            labels = torch.stack([m["masks"].max(0)[0] for m in labels])
            iou_total += (
                torchmetrics.JaccardIndex(
                    task="binary",
                    num_classes=2,
                    average="macro",
                    threshold=0.6,
                )
                .cuda()(outputs, labels)
                .item()
            )

            lab = process_output(labels)
            out = process_output(outputs.softmax(dim=1))
            cm.update_matrix(lab.cpu().int(), out.cpu().int())
        if (not train) and (i >= len(dataloader) - 10):
            image = images[0].permute(1, 2, 0).detach().cpu().numpy()
            lab = output_to_image(labels[0].detach().cpu())
            out = output_to_image(outputs[0].detach().cpu())

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
        "iou": iou_total / data_len,
        "accs": accs,
        "f1_score": f1_score,
    }
    return (run_metrics, output_images, label_images, cm_img)


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
        (metrics, output_images, label_images, cm) = fit_epoch(
            net, trainloader, train_config.learning_rate, train=True, epoch=epoch
        )
        writer.add_scalars(
            f"Train/Metrics/{split}",
            metrics,
            epoch,
        )
        (metrics, output_images, label_images, cm) = fit_epoch(
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
        writer.add_images(
            f"validation_confusion_matrix/{split}", cm, epoch, dataformats="HWC"
        )
        chosen_metric = "f1"
        evaluator = evaluate(net, validationloader, "cuda")
        evaluator.coco_eval["bbox"].stats[-2:]
        evaluator.coco_eval["bbox"].stats[-2:]
        metrics["f1"] = (
            evaluator.coco_eval["bbox"].stats[-2:].prod()
            / evaluator.coco_eval["bbox"].stats[-2:].sum()
        )
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