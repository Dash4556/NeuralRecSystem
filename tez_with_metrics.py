# Creating the class for the model
class YourTezModel(tez.Model):
    # Setting the default function ('__init__') which takes in the user count/song count
    def __init__(self, user_count, song_count):
        super().__init__()

        # Assuming user_count and song_count are the total number of unique users and songs
        self.user_count = user_count
        self.song_count = song_count

        # Adjust the input size accordingly
        input_size = user_count + song_count

        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        # Define the loss function (Mean Squared Error) and optimizer (Adam)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x, y=None):
        # Unpack input x into user_ids and song_ids
        user_ids, song_ids = x

        # Convert user_ids to one-hot encoding if not already in float32 format
        if user_ids.dtype != torch.float32:
            user_ids = F.one_hot(user_ids.to(torch.int64), num_classes=self.user_count).to(torch.float32)

        # Convert song_ids to one-hot encoding if not already in float32 format
        if song_ids.dtype != torch.float32:
            song_ids = F.one_hot(song_ids.to(torch.int64), num_classes=self.song_count).to(torch.float32)

        # Concatenate one-hot encoded user_ids and song_ids along dimension 1
        x = torch.cat([user_ids, song_ids], dim=1)

        # Pass the concatenated tensor through the neural network model
        x = self.model(x)

        # If y is provided (during training), calculate the loss using Mean Squared Error
        if y is not None:
            loss = self.loss_fn(x, y)
            return x, loss
        else:
            # During inference, return the predictions
            return x

    def monitor_metrics(self, output, target):
        # Calculate Root Mean Square Error (RMSE)
        rmse = torch.sqrt(torch.mean((output - target)**2)).item()

        # Convert the output and target to class predictions
        binary_predictions = torch.round(output)

        # Calculate Precision, Recall, and F1_Score
        precision, recall, f1_score, _ = precision_recall_fscore_support(target.cpu().numpy(), binary_predictions.cpu().numpy(), average='weighted')

        # Return a dictionary containing RMSE, Precision, Recall, F1 score
        return {'rmse': rmse, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def fetch_scheduler(self):
        return {}

# Example usage:
# Calculating the number of unique user IDs in the DataFrame
user_count = len(df_final['user_id'].unique())
# Same for number of unique song_ids
song_count = len(df_final['song_id'].unique())

model = YourTezModel(user_count, song_count)

# Continue with creating datasets and training loop as before

# Move the model to the specified device
device = 'cpu'
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        features, target = batch
        # Convert features to tensors
        user_ids = features['user_id'].to(device)
        song_ids = features['song_id'].to(device)
        
        # Convert target to the correct data type
        target = target.to(device, dtype=torch.float32)

        # Forward pass
        predictions, loss = model((user_ids, song_ids), target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation after each epoch
    model.eval()
    val_metrics = {'rmse': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    with torch.no_grad():
        for batch in valid_loader:
            # Convert features and target to tensors
            val_features, val_target = batch
            val_user_ids = val_features['user_id'].to(device)
            val_song_ids = val_features['song_id'].to(device)

            # Forward pass
            val_predictions, val_loss = model((val_user_ids, val_song_ids), val_target)

            # Calculate metrics on the validation set
            val_metrics_batch = model.monitor_metrics(val_predictions, val_target)
            for key in val_metrics.keys():
                val_metrics[key] += val_metrics_batch[key]

    # Normalize metrics by the number of batches
    num_val_batches = len(valid_loader)
    for key in val_metrics.keys():
        val_metrics[key] /= num_val_batches

    # Print metrics after each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Metrics: {val_metrics}')
