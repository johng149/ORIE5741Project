
# Define model
model = T.TransformerModel(
    model_name="all-MiniLM-L6-v2", d_model=512, nhead=8, num_layers=6)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Load data
train_file = "data/train_dataset"
ds = SparkifyDataset(train_file)
dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=mat_collate_fn)
# Assuming `targets` is a torch tensor or numpy array
# You can convert it to a list for easier inspection

emb, pos_indices, target_values, masks = next(iter(dl))
# Print a sample of target values
print(target_values)


print('DONEDONDEDONDEONDEOD')
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for emb, pos_indices, targets, masks in tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()

        outputs = model(emb, mask=masks)
        # Assuming targets are class indices
        loss = loss_function(outputs.transpose(1, 2), targets)

        # Backward pass
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        total_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(dl)}")
