import torch
import os
from model.feature_extractor import FeatureExtractor, VggExtractor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization.images_display import join_images
from model_fitting.configuration import TrainingConfiguration
from torchsummary import summary
from PIL import Image
import torchvision
from model_fitting import losses
from data_loader import augmentation


def fit_epoch(net, feature_extractor: FeatureExtractor, dataloader, lr_rate, train):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    total_loss = 0.0
    output_images = []
    input_images = []
    ref_image = Image.open("starry_night.jpg", mode='r')
    ref_image = augmentation.PairCompose([
        augmentation.ResizeTransform(256),
        augmentation.JPEGcompression(95),
        augmentation.OutputTransform()]
    )(ref_image, 0)[0].unsqueeze(0)
    ref_image = ref_image[..., :256]
    ref_features = feature_extractor(ref_image)
    style_losses = [losses.StyleLoss(f) for f in ref_features]

    for i, data in enumerate(tqdm(dataloader)):
        image, _ = data
        if train:
            optimizer.zero_grad()
        outputs = net(image.cuda())
        features = feature_extractor(outputs)
        input_features = feature_extractor(image)
        content_loss = losses.ContentLoss(input_features[-2])
        c_loss = content_loss(features[-2])
        s_loss = sum(style_losses[i](features[i])
                     for i in range(len(style_losses)))
        loss = c_loss + s_loss/1000
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

        if i >= len(dataloader)-10:
            image = image[0].permute(1, 2, 0).detach().cpu().numpy()
            input_images += [image*256]
            output_images += [outputs[0].permute(1,
                                                 2, 0).detach().cpu().numpy()*256]

    data_len = len(dataloader)
    return total_loss/data_len, input_images, output_images


def fit(net, trainloader, validationloader, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, 'checkpoints.pth')
    checkpoint_conf_path = os.path.join(chp_dir, 'configuration.json')
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    # summary(net, (3, 512, 512))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    images, _ = next(iter(trainloader))

    with torch.no_grad():
        writer.add_graph(net, images[:2].cuda())

    feature_extractor = VggExtractor()
    for epoch in range(train_config.epoch, epochs):
        loss, input_images, output_images = fit_epoch(
            net, feature_extractor, trainloader, train_config.learning_rate, train=True)
        writer.add_scalars('Train/Metrics', {'loss': loss}, epoch)
        grid = join_images(output_images)
        writer.add_images('train_outputs', grid, epoch, dataformats='HWC')
        grid = join_images(input_images)
        writer.add_images('train_inputs', grid, epoch, dataformats='HWC')

        loss, input_images, output_images = fit_epoch(
            net, feature_extractor, validationloader, None, train=False)
        writer.add_scalars('Validation/Metrics', {'loss': loss}, epoch)
        grid = join_images(output_images)
        writer.add_images('validation_outputs', grid, epoch, dataformats='HWC')
        grid = join_images(input_images)
        writer.add_images('validation_inputs', grid, epoch, dataformats='HWC')

        os.makedirs((chp_dir), exist_ok=True)
        if train_config.best_metric > loss:
            train_config.iteration_age = 0
            train_config.best_metric = loss
            print('Epoch {}. Saving model with metric: {}'.format(epoch, loss))
            torch.save(net, checkpoint_name_path.replace('.pth', '_final.pth'))
        else:
            train_config.iteration_age += 1
            print('Epoch {} metric: {}'.format(epoch, loss))
        if train_config.iteration_age == lower_learning_period:
            train_config.learning_rate *= 0.5
            train_config.iteration_age = 0
            print("Learning rate lowered to {}".format(
                train_config.learning_rate))
        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(net.state_dict(), checkpoint_name_path.replace(
            '.pth', '_final_state_dict.pth'))
    print('Finished Training')
