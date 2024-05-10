import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc1_2 = nn.Linear(768, 512)
        self.fc1_3 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1_2 = nn.BatchNorm1d(512)
        self.bn1_3 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.relu1_2 = nn.ReLU()
        self.relu1_3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # Added dropout
        self.fc2 = nn.Linear(512, 256)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc2_3 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn2_2 = nn.BatchNorm1d(256)
        self.bn2_3 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.relu2_3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)  # Adjusted dropout
        # Residual connection layer
        self.fcRes = nn.Linear(256*3, 256*3)  # Ensuring the residual layer doesn't change dimension
        self.bnRes = nn.BatchNorm1d(256*3)
        self.reluRes = nn.ReLU()
        self.fc3 = nn.Linear(256*3, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(128, 16)

    def forward(self, x, x_2, x_3):
        x = x.view(-1, 768)
        x_2 = x_2.view(-1, 768)
        x_3 = x_3.view(-1, 768)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x_2 = self.dropout1(self.relu1_2(self.bn1_2(self.fc1_2(x_2))))
        x_3 = self.dropout1(self.relu1_3(self.bn1_3(self.fc1_3(x_3))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x_2 = self.dropout2(self.relu2_2(self.bn2_2(self.fc2_2(x_2))))
        x_3 = self.dropout2(self.relu2_3(self.bn2_3(self.fc2_3(x_3))))
        x = torch.cat((x, x_2, x_3), dim=1)
        identity = x
        x = self.fcRes(x)
        x = self.bnRes(x)
        x = self.reluRes(x) + identity  # Add residual connection
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.relu4(self.bn4(self.fc4(x)))
        x = self.relu5(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return torch.softmax(x, dim=1)

class CustomDataset:
    def __init__(self, data_path):
        # Load data from npz files
        self.dash = np.load(data_path + '/total_dashboard_data.npz')['embeddings']
        self.rear = np.load(data_path + '/total_rearview_data.npz')['embeddings']
        self.side = np.load(data_path + '/total_side_data.npz')['embeddings']
        self.labels = np.load(data_path + "/total_labels.npz")['labels']

        # Identify indices where labels are 0 or 1. This, as the nature of the dataset is to have both 0 and 1 as the same class.
        class_0_and_1_indices = np.where((self.labels == 0) | (self.labels == 1))[0]

        # Select every nth index from class 0 and 1 indices
        thinned_indices = class_0_and_1_indices[::1]

        # Get indices of other classes
        other_class_indices = np.where((self.labels != 0) & (self.labels != 1))[0]

        # Combine indices to keep
        keep_indices = np.sort(np.concatenate((thinned_indices, other_class_indices)))

        # Filter datasets to keep only selected indices
        self.dash = self.dash[keep_indices]
        self.rear = self.rear[keep_indices]
        self.side = self.side[keep_indices]
        self.labels = self.labels[keep_indices]
        # Merge class 1 into class 0
        self.labels = np.where(self.labels == 1, 0, self.labels)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Generate embeddings and label for a given index
        embedding = (torch.Tensor(self.dash[idx]), torch.Tensor(self.rear[idx]), torch.Tensor(self.side[idx]))
        label = self.labels[idx] - 1 if self.labels[idx] > 0 else self.labels[idx]
        return (*embedding, torch.Tensor([label]).long())

    def get_class_counts(self):
        # Compute unique labels and their counts
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return unique_labels, counts

class CustomDatasetTest:
    def __init__(self, data_path):
        # Load data from npz files
        self.dash = np.load(data_path + '/total_dashboard_data.npz')['embeddings']
        self.rear = np.load(data_path + '/total_rearview_data.npz')['embeddings']
        self.side = np.load(data_path + '/total_side_data.npz')['embeddings']
        self.labels = np.load(data_path + "/total_labels.npz")['labels']

        # Identify indices where labels are 0 or 1
        class_0_and_1_indices = np.where((self.labels == 0) | (self.labels == 1))[0]

        # Select every 10th index from class 0 and 1 indices
        thinned_indices = class_0_and_1_indices[::1]

        # Get indices of other classes
        other_class_indices = np.where((self.labels != 0) & (self.labels != 1))[0]

        # Combine indices to keep
        keep_indices = np.sort(np.concatenate((thinned_indices, other_class_indices)))

        # Filter datasets to keep only selected indices
        self.dash = self.dash[keep_indices]
        self.rear = self.rear[keep_indices]
        self.side = self.side[keep_indices]
        self.labels = self.labels[keep_indices]
        # Merge class 1 into class 0
        self.labels = np.where(self.labels == 1, 0, self.labels)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Generate embeddings and label for a given index
        embedding = (torch.Tensor(self.dash[idx]), torch.Tensor(self.rear[idx]), torch.Tensor(self.side[idx]))
        label = self.labels[idx] - 1 if self.labels[idx] > 0 else self.labels[idx]
        return (*embedding, torch.Tensor([label]).long())

    def get_class_counts(self):
        # Compute unique labels and their counts
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return unique_labels, counts





# Instantiate the neural network
model = CustomNet().cuda()



# Create custom dataset and dataloader
data_dir = 'clipData/dataset/'  # Directory containing the combined .npz files for each view
test_dir = 'clipData/trainTest/'  # Directory containing the combined .npz files for each view
#labels_file = 'path_to_your_labels_file.npz'  # File containing the labels
dataset = CustomDataset(data_dir)
testDataset = CustomDatasetTest(test_dir)
# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders for train and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=128*5, shuffle=True, num_workers=31)
val_dataloader = DataLoader(val_dataset, batch_size=128*5, shuffle=False, num_workers=31)
test_dataloader = DataLoader(testDataset, batch_size=128*5, shuffle=False, num_workers=31)


unique_labels, class_counts = dataset.get_class_counts()
print("Unique labels:", unique_labels)
print("Class counts:", class_counts)

total = sum(class_counts)
class_weights = [total / class_counts[i] for i in range(len(class_counts))]
class_weights_tensor = torch.FloatTensor(class_weights).cuda()
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=1e-4) #1e-5 is a good starting point



num_epochs = 100

# Learning rate scheduler
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

# Add early stopping parameters
patience = 3  # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0  # Counter for epochs without improvement

# Training loop

best_val_loss = float('inf')
best_val_acc = 0
best_test_acc = 0
best_test_epoch = 0
best_val_epoch = 0

from tqdm import tqdm
model_path = 'clipData/bestModel.pth'  # Replace with the path where you want to save the model
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (inputs1, inputs2, inputs3, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs1 = inputs1.cuda()  # Move inputs to GPU
        inputs2 = inputs2.cuda()
        inputs3 = inputs3.cuda()
        labels = labels.cuda()  # Move labels to GPU
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs1, inputs2, inputs3)

        # Compute loss
        loss = criterion(outputs, labels.squeeze())  # Squeeze labels to remove extra dimension

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if i % 100 == 99:
            #print('[Train] [%d, %5d] loss: %.3f' %
            #      (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    test_acc = 0
    with torch.no_grad():
        for inputs1, inputs2, inputs3, labels in val_dataloader:
            inputs1 = inputs1.cuda()  # Move inputs to GPU
            inputs2 = inputs2.cuda()
            inputs3 = inputs3.cuda()
            labels = labels.cuda()  # Move labels to GPU
            outputs = model(inputs1, inputs2, inputs3)
            _, predicted = torch.max(outputs, 1)
            #print('predictions',predicted)
            #print('labels',labels.squeeze())
            loss = criterion(outputs, labels.squeeze())
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
        #input("Press Enter to continue...")

    # Adjust learning rate based on validation loss
    val_loss /= len(val_dataloader)
    scheduler.step(val_loss)

    # If this epoch's validation loss is the best so far, save the model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch
        #torch.save(model, model_path)  # Replace with the path where you want to save the model
        #torch.save(model.state_dict(), '/home/cvrr/Desktop/CVPRchallenge/CVPRchallenge/clipData/bestStateDict2.pth')
        #print('Model saved')
        
    if best_val_acc < 100 * correct / total:
        best_val_acc = 100 * correct / total    
        epochs_no_improve = 0  # Reset the counter
    else:
        epochs_no_improve += 1  # Increment the counter

    # If we've waited for patience epochs without improvement, stop training
    if epochs_no_improve == patience:
        print('Early stopping after %d epochs without improvement' % patience)
        break


    #print('[Epoch %d] Loss: %.4f | Validation loss: %.3f | Accuracy: %.2f %%' %
    #      (epoch + 1, loss, val_loss, 100 * correct / total))
    print(f'[Epoch {epoch}] Loss {loss}, Validation loss {val_loss}, Validation Accuracy {100 * correct / total}, current best: {best_val_acc} at epoch {best_val_epoch}')
    print('Current learning rate:', optimizer.param_groups[0]['lr'])
#print('Finished Training')
    #print('Testing')
    # Test loop
    #model = torch.load(model_path)
    #model = model.cuda()  # Move model to GPU
    #model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    # Initialize lists to store the true and predicted labels
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for input1, input2, input3, labels in test_dataloader:
            input1 = input1.cuda()  # Move inputs to GPU
            input2 = input2.cuda()
            input3 = input3.cuda()
            labels = labels.cuda()  # Move labels to GPU
            outputs = model(input1, input2, input3)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

            # Append the true and predicted labels to the lists
            true_labels.append(labels.squeeze().cpu().numpy())
            pred_labels.append(predicted.cpu().numpy())
    test_acc = 100 * correct / total
    if test_acc > best_test_acc:
        best_test_acc = 100 * correct / total
        best_test_epoch = epoch
        torch.save(model, model_path)  # Replace with the path where you want to save the model
        torch.save(model.state_dict(), 'clipData/bestStateDict2.pth')
        print('Model saved')
        import pandas as pd
        # Create a DataFrame from the true and predicted labels
        df = pd.DataFrame({
            'True Labels': np.concatenate(true_labels),
            'Predicted Labels': np.concatenate(pred_labels)
        })
        # Save the DataFrame as a CSV file, overwriting any existing file
        df.to_csv('data/examples/labels_and_predictions_2.csv', index=False)

        # Compute the confusion matrix
        cm = confusion_matrix(np.concatenate(true_labels), np.concatenate(pred_labels))

        # Normalize the confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Round the confusion matrix values to two decimal places
        cm = np.around(cm, decimals=2)

        # Increase figure size for better visibility
        fig, ax = plt.subplots(figsize=(20, 20))  # You can adjust the size as needed

        # Create the confusion matrix display object
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(16))

        # Use larger font for the numbers inside the boxes
        plt.rcParams.update({'font.size': 16})  # Adjust font size as needed

        # Plot the confusion matrix with color map
        disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax)

        # Save the confusion matrix as an image
        plt.savefig('data/examples/confusion_matrix_2.png', bbox_inches='tight')
        plt.close()
    #print('Test Accuracy: %.2f %%' % (100 * correct / total))
    print(f'Test Accuracy {test_acc}, current best: {best_test_acc} at epoch {best_test_epoch}')
    scheduler.step(test_acc)