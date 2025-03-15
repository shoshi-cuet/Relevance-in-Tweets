import torch
import torch.nn as nn
import copy
from tqdm import tqdm

class LSTMClassifier(nn.Module):
    """
    A PyTorch implementation of an LSTM-based classifier for text classification.

    Attributes:
        embedding_dim (int): The size of the word embeddings.
        hidden_dim (int): The size of the hidden layer in the LSTM.
        output_dim (int): The size of the output layer.
        dropout (float): The dropout rate.
        pretrained_embeddings (torch.Tensor): Pretrained embeddings to initialize the embedding layer.

    Methods:
        forward(text): Processes the input text and produces output logits for classification.
        fit(dataloaders, optimizer, criterion, num_epochs, patience): Trains the model using the provided dataloaders, optimizer, and loss criterion for a specified number of epochs and early stopping patience.
        predict(dataloader): Generates predictions using the trained model on the provided dataloader.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout, pretrained_embeddings):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden.squeeze(0))
        output = self.fc(hidden)
        return output

    def fit(self, dataloaders, optimizer, criterion, num_epochs, patience,device):
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