import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize


class MaskDetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32*61*61, 1952),
            nn.ReLU(),
            nn.Linear(1952, 488),
            nn.ReLU(),
            nn.Linear(488, 2))

    def forward(self, xb):
        return self.network(xb)


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


if __name__ == '__main__':

    data_dir = 'Testset'  # Name of the folder containing data folder
    model = MaskDetectorModel()
    model.load_state_dict(torch.load('Mask.pth'))
    transform = transforms.Compose([Resize((244, 244)), ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)

    for data in dataset:
        img, label = data
        res = predict_image(img, model)
        print("Mask is detected" if res == 0 else "Mask not detetcted")
