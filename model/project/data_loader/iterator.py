import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset
import random

class StringDatasetTrain(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)**2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        first_id = idx%len(self.data)
        second_id = idx//len(self.data)
        if 0.5 > random.random():
            second_id = first_id
        fist_set, second_set = self.data[first_id], self.data[second_id]
        fist_text, second_text = random.choice(fist_set), random.choice(second_set)
        if len(fist_text)==0 or len(second_text)==0:
            print(fist_text), print(second_text)
        if self.transform:
            fist_text = self.transform(fist_text)
            second_text = self.transform(second_text)
        sample  = {
            'first_text': torch.FloatTensor(fist_text), 
            'second_text': torch.FloatTensor(second_text), 
            'loss_mul': 1., 
            'label': float(first_id==second_id)
        }

        return sample

class StringDatasetValidation(Dataset):
    def __init__(self, data, leagues, transform=None):
        self.transform = transform
        self.data_values = []
        for key in data:
            for val in data[key]:
                league_key = ','.join(key.split(',')[:-1])
                label = key.split(',')[-1]
                if league_key in leagues:
                    self.data_values.append((val,label, leagues[league_key]))
        self.data_values = [(i, l, x) for i, l, x in self.data_values if i not in x and l in x]
        self.leagues = leagues

    def __len__(self):
        return len(self.data_values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        team, label, posible_matches,  = self.data_values[idx]
        team_name, posible_matches_names = team, posible_matches
        label = posible_matches.index(label)
        if self.transform:
            team = self.transform(team)
            posible_matches = [self.transform(x) for x in posible_matches]
        sample  = {
            'team': torch.FloatTensor(team), 
            'posible_matches': torch.FloatTensor(posible_matches), 
            'label': label,
            'team_name': team_name,
            'posible_matches_names':posible_matches_names,
        }

        return sample