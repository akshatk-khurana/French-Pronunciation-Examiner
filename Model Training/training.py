import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SiameseNetwork()

model = model.to(device)

learning_rate = 0.001

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 100
for epoch in range(epochs):
    if epoch == 75:
      learning_rate = 0.0005
    model.train()
    running_loss = 0.0

    for standard, spoken, labels in train_loader:
        standard = standard.to(device)
        spoken = spoken.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(standard.to(device), spoken.to(device))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Train] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

model.to('cpu')
model.eval()
torch.save(model.state_dict(), gdrive_path + 'Siamese_Model.pth')