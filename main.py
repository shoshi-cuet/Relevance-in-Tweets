import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import preprocess
from models.CNN import CNN
import numpy as np
import argparse
from models.LSTM import LSTMClassifier
from helpers import *

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    if args.subtask =='a':  
        train_df = pd.read_csv('subtaskA/CT22_english_1A_checkworthy_train.tsv', sep='\t')
        test_df  = pd.read_csv('subtaskA/CT22_english_1A_checkworthy_dev_test.tsv',sep='\t')
    elif args.subtask =='b':
        train_df = pd.read_csv('subtaskB/CT22_english_1B_claim_train.tsv', sep='\t')
        test_df  = pd.read_csv('subtaskB/CT22_english_1B_claim_dev_test.tsv',sep='\t')

    # imbalanced data set so calculating the class weights
    class_weight_0, class_weight_1 = calculate_class_weights(train_df)

    # Create a weight tensor
    weights = torch.tensor([class_weight_0, class_weight_1])
    weights = weights.to(device)

    # Load the pre-trained word embeddings
    glove_file = 'glove/glove.6B.50d.txt'
    glove_embeddings = load_glove_embeddings(glove_file)

    # Create a matrix of embeddings for words in the vocabulary
    word_to_index = {word: index for index, word in enumerate(glove_embeddings.keys())}
    embedding_dim = 50
    embedding_matrix = create_embedding_matrix(word_to_index, glove_embeddings, embedding_dim)

    # Define the hyperparameters for both models
    num_epochs = args.num_epochs
    patience = args.patience
    output_dim = 2
    dropout = args.dropout
    lr = args.lr

    # Preprocess the data and split into input and target
    inputs = [text_to_word_indices(preprocess(text), word_to_index) for text in train_df['tweet_text']]
    targets = train_df['class_label'].values

    # Preprocess the test data and split into input and target
    test_inputs = [text_to_word_indices(preprocess(text), word_to_index) for text in test_df['tweet_text']]
    test_targets = test_df['class_label'].values

    # Pad the inputs with zeros
    padded_inputs = pad_inputs(inputs)
    padded_test_inputs = pad_inputs(test_inputs)

    X = torch.LongTensor(padded_inputs)
    y = torch.LongTensor(np.array(targets))

    # Convert the test data to PyTorch tensors
    X_test = padded_test_inputs
    y_test = torch.LongTensor(np.array(test_targets))

    # split into train and validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


    dataloaders = {'train':DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True),
                   'validation':DataLoader(list(zip(X_valid, y_valid)), batch_size=32, shuffle=False),
                   'test':DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)}

    # Create the pre-trained embeddings tensor
    pretrained_embeddings = torch.tensor(embedding_matrix, dtype=torch.float)

    if args.model_type =='LSTM':
        hidden_dim = args.hidden_dim
        
        model = LSTMClassifier( embedding_dim=embedding_dim, 
                                hidden_dim=hidden_dim, 
                                output_dim=output_dim, 
                                dropout=dropout, 
                                pretrained_embeddings=pretrained_embeddings)

    elif args.model_type =='CNN':
        num_filters = args.num_filters
        filter_sizes = args.filter_sizes
        hidden_dim = args.hidden_dim

        model = CNN(embedding_dim = embedding_dim, 
                    num_filters=num_filters, 
                    filter_sizes=filter_sizes,
                    hidden_dim = hidden_dim, 
                    output_dim=output_dim, 
                    dropout=dropout, 
                    pretrained_embeddings=pretrained_embeddings)


    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.fit(dataloaders=dataloaders, optimizer=optimizer, criterion=criterion, num_epochs=num_epochs,patience=patience,device=device)


#############TEST##############################################################

    state_dict = torch.load('best_model_weights.pth')
    model.load_state_dict(state_dict)

    # Get the model's predictions on the test data
    test_preds = model.predict(dataloaders['test'],device=device)

    # Evaluate the model's performance on the test data
    scores = evaluate(test_targets, test_preds, args)
    return scores

if __name__ == "__main__":
    # Create argument parser for LSTM model
    lstm_parser = argparse.ArgumentParser(description="Arguments for LSTM model.")
    lstm_parser.add_argument("--subtask", type=str, default="a", choices=["a", "b"], help="Choose the subtask by default it is a")
    lstm_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate. Default is 1e-5.")
    lstm_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. Default is 0.2.")
    lstm_parser.add_argument("--hidden_dim", type=int, default=20, help="Hidden dimension size for LSTM. Default is 20.")
    lstm_parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs. Default is 10.")
    lstm_parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping. Default is 3.")

    # Create argument parser for CNN model
    cnn_parser = argparse.ArgumentParser(description="Arguments for CNN model.")
    cnn_parser.add_argument("--subtask", type=str, default="a", choices=["a", "b"], help="Choose the subtask by default it is a")
    cnn_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Default is 1e-4.")
    cnn_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate. Default is 0.5.")
    cnn_parser.add_argument("--filter_sizes", type=str, default=[3,4,5], help="Filter sizes for CNN. Default is '3,4,5'.")
    cnn_parser.add_argument("--num_filters", type=int, default=100, help="Number of filters for CNN. Default is 100.")
    cnn_parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs. Default is 10.")
    cnn_parser.add_argument("--hidden_dim", type=int, default=20, help="Hidden dimension size for CNN. Default is 20.")
    cnn_parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping. Default is 3.")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Choose between LSTM and CNN models for text classification.")
    parser.add_argument("--model_type", type=str, default="LSTM", choices=["LSTM", "CNN"], help="Choose the model type (LSTM or CNN). Default is LSTM.")
    args, _ = parser.parse_known_args()

    if args.model_type == "LSTM":
        model_parser = lstm_parser
    elif args.model_type == "CNN":
        model_parser = cnn_parser
    else:
        raise ValueError("Invalid model type.")

    # Parse model-specific command-line arguments
    model_args, _ = model_parser.parse_known_args()

    # Merge arguments
    args = argparse.Namespace(**vars(args), **vars(model_args))

    # Call main function
    main(args)