from torch import nn
import torch
from tqdm import tqdm


def fit_epoch(net, dataloader, lr_rate, train, epoch=1):
    if train:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    else:
        net.eval()
    criterion = nn.MSELoss()
    torch.set_grad_enabled(train)
    torch.backends.cudnn.benchmark = train
    losses = 0.0
    for i, data in enumerate(tqdm(dataloader)):
        cur_state, next_states, act, _, _ = data
        if train:
            optimizer.zero_grad()
        outputs = net(torch.cat((cur_state, act)))
        loss = criterion(outputs, next_states - cur_state)
        if train:
            loss.backward()
            optimizer.step()
        losses += loss.item()
    data_len = len(dataloader)
    return losses/data_len
