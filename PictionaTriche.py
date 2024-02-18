import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ssl
from PIL import Image
import torchvision.transforms.functional as TF
import gradio as gr
import os
from PIL import Image

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = x.reshape(-1, 16 * 5 * 5)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class MyApp:
    def __init__(self):
        super().__init__()
        self.epoch = 100
        self.lr = 0.001
        self.model = MyModel()

        self.classes = ('squares', 'circles', 'triangles')
        self.labels = (0, 1, 2)

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        self.dataset = torchvision.datasets.ImageFolder(root='shapes', transform=self.transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=2)

app = MyApp()

def train_model(device):
    loss_fonct = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(app.model.parameters(), lr=0.01)

    train_accuracies = np.zeros(app.epoch)
    train_loss = []

    for epoch in tqdm(range(app.epoch)):
        total_train, correct_train = 0, 0
        for batch_idx, batch in enumerate(tqdm(app.dataloader)):
            images, labels = batch
            images = images.to(device=device)
            labels = labels.to(device=device)

            output = app.model.forward(images)
            loss = loss_fonct(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                train_loss.append(loss.item())

        train_accuracies[epoch] = correct_train / total_train * 100
        print(correct_train / total_train * 100, "%\n")

    torch.save(app.model.state_dict(), "shapes-model.pth")

def predict(image):
    out = app.model(image.reshape(1, 3, 32, 32))
    _, pred = torch.max(out, dim=1)
    return app.classes[pred.item()]

def np_array_to_tensor_image(img, width=32, height=32, device='cpu'):
    image = Image.fromarray(img).convert('RGB').resize((width, height))
    image = transforms.Compose([
        transforms.ToTensor(),
    ])(image).unsqueeze(0)
    return image.to(device, torch.float)

def sketch_recognition(img):
    img = np_array_to_tensor_image(img)
    app.model.eval()
    with torch.no_grad():
        result = predict(img)
    app.model.train()
    return result

ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app.model.to(device)

# if os.path.isfile("shapes-model.pth"):
#     app.model.load_state_dict(torch.load("shapes-model.pth"))
# else:
    # train_model(device)
train_model(device)
gr.Interface(fn=sketch_recognition, inputs=["sketchpad"], outputs="label").launch(share=True)
