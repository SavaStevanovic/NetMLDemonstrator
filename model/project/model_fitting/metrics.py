from sklearn.metrics import f1_score, accuracy_score
import torch

def metrics( net, dataloader, epoch=1):
    net.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs1, inputs2, _, label = data['first_text'], data['second_text'], data['loss_mul'], data['label']
            inputs1_cuda, inputs2_cuda = inputs1.to('cuda'), inputs2.to('cuda')
            
            outputs1, outputs2 = net(inputs1_cuda, inputs2_cuda)
            loss = (outputs2 - outputs1).pow(2).sum(1)
            predicted = loss < 0.1
            predictions+=predicted.cpu().numpy().tolist()
            labels+=label.numpy().tolist()
    print("Class balance {}".format(sum(labels)/len(labels)))
    return f1_score(labels, predictions), accuracy_score(labels, predictions)

def validation_metrics(net, dataloader, epoch=1):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs1, posible_targets, label= data['team'], data['posible_matches'], data['label']
            team_name, posible_matches_names = data['team_name'], data['posible_matches_names']
            inputs1_cuda = inputs1.cuda()
            min_loss = 10000
            ind = -1
            closeness = []
            for i, x in enumerate(posible_targets.squeeze()):
                outputs1, outputs2 = net(inputs1_cuda, x.unsqueeze(0).cuda())
                loss = (outputs2 - outputs1).pow(2).sum(1)
                closeness.append((posible_matches_names[i][0], loss.cpu().numpy()[0]))
                if loss< min_loss:
                    min_loss = loss
                    ind = i
            closeness = sorted(closeness, key=lambda tup: tup[1])    
            correct += ind==label
            if ind!=label:
                print("For '{}' predicted match was '{}' but correct is '{}'".format(team_name[0], posible_matches_names[ind][0], posible_matches_names[label][0]))
                print(closeness)
                print("-----------------------------------------")
    return correct.numpy()[0]/len(dataloader)