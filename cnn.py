import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

# ==== process ====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的裝置為: {device}")
        

class CardClassifier(nn.Module):
    def __init__(self, num_classes=13):  # 13種數字
        super(CardClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f"無法讀取圖片：{path}，錯誤：{e}")
            # 回傳一張空白圖（或你可以選擇 return None）
            sample = Image.new('RGB', (224, 224))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.tensor(target)
    
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

dataset = SafeImageFolder('dataset/', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = CardClassifier(num_classes=13).to(device)
# ==== end ====


# ==== train ====

def train_model():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        running_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}")

    # ==== save ====
    torch.save(model.state_dict(), 'card_classifier.pth')
    print("模型權重已儲存為 card_classifier.pth")
    # ==== end ====
# ==== end ====

if __name__ == "__main__":
    train_model()