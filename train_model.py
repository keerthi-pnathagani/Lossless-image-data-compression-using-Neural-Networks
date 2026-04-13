import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,3,3,padding=1)
        )

    def forward(self,x):
        return self.net(x)

model = Predictor().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 5   # keep small for now

for epoch in range(epochs):
    for images, _ in loader:
        images = images.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

torch.save(model.state_dict(), "model.pth")

print("✅ Training complete. Model saved as model.pth")
