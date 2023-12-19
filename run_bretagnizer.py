import argparse

import src.constants as const
from src import multilayer_perceptron as mlp
from src.utils import load_inputs, get_vocab_and_size, get_mappings, build_n_grams, get_batch_idx


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
