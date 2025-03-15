import torch
import torch.nn as nn
import copy
from tqdm import tqdm

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for text classification.

    Args:
        embedding_dim (int): The size of the word embeddings.
        num_filters (int): The number of filters for each convolution layer.
        filter_sizes (List[int]): A list of filter sizes for each convolution layer.
        hidden_dim (int): The size of the hidden layer.
        output_dim (int): The size of the output layer.
        dropout (float): The dropout rate.
        pretrained_embeddings (torch.Tensor): Pretrained embeddings to initialize the embedding layer.
    """
    def __init__(self, embedding_dim, num_filters, filter_sizes, hidden_dim, output_dim, dropout, pretrained_embeddings):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.vocab = pretrained_embeddings
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = self.relu(conv_layer(embedded))
            pool_output = nn.functional.max_pool1d(conv_output, conv_output.shape[2]).squeeze(2)
            conv_outputs.append(pool_output)

        concat_output = torch.cat(conv_outputs, dim=1)
        fc1_output = self.dropout(self.relu(self.fc1(concat_output)))
        fc2_output = self.fc2(fc1_output)
        return fc2_output


    def fit(self, dataloaders, optimizer, criterion, num_epochs, patience, device):
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = float('inf')

        early_stopping_counter = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')

            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(torch.long).to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | {phase.capitalize()} Acc: {epoch_acc:.4f}')

                if phase == 'validation':
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.state_dict())
                        torch.save(best_model_wts, 'best_model_weights.pth')  # Save the best model weights to a file
                        early_stopping_counter = 0

                    else:
                        early_stopping_counter += 1
                        print(f'Early stopping counter: {early_stopping_counter} out of {patience}')
                        if early_stopping_counter >= patience:
                            print('Early stopping')
                            self.load_state_dict(best_model_wts)
                            return

    def predict(self, dataloader,device):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(torch.long).to(device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().tolist())
        return predictions
