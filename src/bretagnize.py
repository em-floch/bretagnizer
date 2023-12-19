import pandas as pd
import constants as const
import random
import argparse
import multilayer_perceptron as mlp


def load_inputs(input_path):
    """Load the inputs from the csv file into a list of names"""
    df = pd.read_csv(input_path, sep=";")
    inputs = list(df[const.COMMUNE_NAME_COL].values)
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


def run_bretagnizer(input_path,
                    block_size,
                    hidden_size,
                    learning_rate,
                    epochs,
                    batch_size,
                    embedding_size):
    inputs = load_inputs(input_path)
    voc, voc_size = get_vocab_and_size(inputs)
    char_to_idx, idx_to_char = get_mappings(voc)
    train_x, train_y = build_n_grams(inputs, char_to_idx, n=3)

    model = mlp.MultilayerPerceptron(
        learning_rate=learning_rate,
        block_size=block_size,
        hidden_size=hidden_size,
        len_vocab=voc_size,
        embedding_size=embedding_size
    )

    for epoch in range(epochs):
        train_batch_x, train_batch_y = get_batch_idx(train_x, train_y, batch_size)
        logits, loss = model.forward(train_batch_x, train_batch_y)
        model.backward(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss}")
            for _ in range(10):
                print(model.sample_from_distribution(idx_to_char))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/communes-france.csv")
    parser.add_argument("--block_size", type=str, default=const.BLOCK_SIZE)
    parser.add_argument("--hidden_size", type=str, default=const.HIDDEN_SIZE)
    parser.add_argument("--learning_rate", type=str, default=const.LEARNING_RATE)
    parser.add_argument("--epochs", type=str, default=const.EPOCHS)
    parser.add_argument("--batch_size", type=str, default=const.BATCH_SIZE)
    parser.add_argument("--embedding_size", type=str, default=const.EMBEDDING_SIZE)

    args = parser.parse_args()
    run_bretagnizer(args.input_path,
                    args.block_size,
                    args.hidden_size,
                    args.learning_rate,
                    args.epochs,
                    args.batch_size,
                    args.embedding_size)
