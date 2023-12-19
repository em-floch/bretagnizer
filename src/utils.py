import pandas as pd
import random


def load_inputs(input_path, col_name):
    """Load the inputs from the csv file into a list of names"""
    df = pd.read_csv(input_path, sep=";")
    inputs = list(df[col_name].values)
    return inputs

def get_vocab_and_size(input_texts):
    """Get the vocabulary and its size from the input list of names"""
    vocab = "".join(input_texts)
    vocab = ['.'] + list(sorted(set(vocab)))
    vocab_size = len(vocab)
    return vocab, vocab_size


def get_mappings(vocab):
    """Get the mappings from characters to indices and vice-versa"""
    char_to_idx = {char: i for i, char in enumerate(vocab)}
    idx_to_char = {i: char for i, char in enumerate(vocab)}
    return char_to_idx, idx_to_char

def add_dots(input, n):
    """Add dots at the beginning and end of each name"""
    return "." * n + input + "."

def build_n_grams(inputs, char_to_idx, n):
    """Build n-grams from the input list of names. Return the n-grams and the next character in
    two lists"""
    n_grams_x = []
    y = []
    for input in inputs:
        input = add_dots(input, n)
        for i in range(len(input) - n):
            n_gram = input[i:i + n]
            print(n_gram, input[i + n])
            n_grams_x.append([char_to_idx[char] for char in n_gram])
            y.append(char_to_idx[input[i + n]])
    return n_grams_x, y


def get_batch_idx(train_x, train_y, batch_size):
    """ Get a random batch of indices from the training data"""
    batch_idx = random.sample(range(len(train_x)), batch_size)
    train_batch_x = [train_x[i] for i in batch_idx]
    train_batch_y = [train_y[i] for i in batch_idx]
    return train_batch_x, train_batch_y

