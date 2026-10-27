import torch
from torch import nn, optim
from tqdm import tqdm

from config import Config
from models.simple_net import SimpleNet
from utils.data_utils import get_dataloaders

cfg = Config()

device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

train_loader, test_loader = get_dataloaders(cfg.batch_size)
model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, cfg.num_epochs + 1):
    print(f"\n🌟 Epoch {epoch}/{cfg.num_epochs}")
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100.0 * correct / total
print(f"\n📊 Test accuracy: {test_accuracy:.2f}%")

torch.save(model.state_dict(), "mnist_model.pt")
print("\n✅ Training complete! Model saved as mnist_model.pt")
