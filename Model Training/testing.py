import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(gdrive_path + "Siamese_Model.pth", map_location=device))
criterion = nn.BCELoss()

num_epochs = 5

for epoch in range(num_epochs):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for standard, spoken, labels in test_loader:
            standard = standard.to(device)
            spoken = spoken.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(standard, spoken)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    print(f"[Test] Epoch [{epoch+1}/{num_epochs}], Loss: {test_loss/len(test_loader):.4f}")