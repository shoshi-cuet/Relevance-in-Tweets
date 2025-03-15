import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple

def calculate_class_weights(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate class weights for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'class_label' column.

    Returns:
        Tuple[float, float]: The calculated class weights.
    """
    total_samples = len(df)
    num_classes = df['class_label'].nunique()

    class_samples_0 = len(df[df['class_label'] == 0])
    class_samples_1 = total_samples - class_samples_0

    class_weight_0 = total_samples / (num_classes * class_samples_0)
    class_weight_1 = total_samples / (num_classes * class_samples_1)

    return class_weight_0, class_weight_1

def pad_inputs(inputs: List[List[int]]) -> torch.Tensor:
    """
    Pad the input sequences with zeros.

    Args:
        inputs (List[List[int]]): The input sequences.

    Returns:
        torch.Tensor: The padded input sequences.
    """
    inputs = [torch.tensor(input_indices) for input_indices in inputs]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_inputs

def evaluate(true_labels: List[int], predicted_labels: List[int], args) -> Dict[str, float]:
    """
    Evaluate the performance of the model.

    Args:
        true_labels (List[int]): The true labels.
        predicted_labels (List[int]): The predicted labels.
        args: The arguments containing 'subtask' and 'model_type'.

    Returns:
        Dict[str, float]: A dictionary with evaluation metrics.
    """
    scores = {
        'Precision': precision_score(true_labels, predicted_labels),
        'Recall' : recall_score(true_labels, predicted_labels),
        'F1' : f1_score(true_labels, predicted_labels),
        'Auc_roc' : roc_auc_score(true_labels, predicted_labels),
        'Accuracy' : accuracy_score(true_labels, predicted_labels)
    }
    
    print(f'Results for subtask: {args.subtask}, with model type: {args.model_type}\n')
    print('-'*25)
    print(scores)
    return scores

    
def load_glove_embeddings(embedding_file: str) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from a file.

    Args:
        embedding_file (str): The path to the GloVe embeddings file.

    Returns:
        Dict[str, np.ndarray]: A dictionary with words and their corresponding embeddings.
    """
    embeddings_index = {}
    with open(embedding_file) as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    return embeddings_index

def create_embedding_matrix(word_to_index: Dict[str, int], embeddings_dict: Dict[str, np.ndarray], embedding_dim: int) -> np.ndarray:
    """
    Create an embedding matrix for the given vocabulary.

    Args:
        word_to_index (Dict[str, int]): A dictionary with words and their corresponding indices.
        embeddings_dict (Dict[str, np.ndarray]): A dictionary with words and their corresponding embeddings.
        embedding_dim (int): The dimension of the embeddings.

    Returns:
        np.ndarray: The embedding matrix.
    """
    embedding_matrix = np.zeros((len(word_to_index), embedding_dim))
    
    for word, i in word_to_index.items():
        if word not in embeddings_dict:
            embedding = np.random.normal(scale=0.6, size=(embedding_dim,))
            embeddings_dict[word] = embedding
        else:
            embedding = embeddings_dict[word]
        
        embedding_matrix[i] = embedding
    
    return embedding_matrix

def text_to_word_indices(text: str, word_to_index: Dict[str, int]) -> List[int]:
    """
    Convert a text to a list of word indices based on the provided mapping.

    Args:
        text (str): The input text.
        word_to_index (Dict[str, int]): A dictionary with words and their corresponding indices.

    Returns:
        List[int]: A list of word indices.
    """
    words = text.split()
    indices = [word_to_index[word] for word in words if word in word_to_index]
    return indices