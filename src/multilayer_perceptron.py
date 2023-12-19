import torch


class MultilayerPerceptron:

    def __init__(self, embedding_size, block_size, hidden_size, len_vocab, learning_rate):

        self.learning_rate = learning_rate
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.len_vocab = len_vocab
        self.embedding_size = embedding_size
        self.embedding = torch.randn((self.len_vocab, self.embedding_size), requires_grad=True)

        self.w1 = torch.randn((self.embedding_size * self.block_size, self.hidden_size), requires_grad=True)
        self.b1 = torch.randn(self.hidden_size, requires_grad=True)
        self.w2 = torch.randn((self.hidden_size, self.len_vocab), requires_grad=True)
        self.b2 = torch.randn(self.len_vocab, requires_grad=True)
        self.params = [self.w1, self.b1, self.w2, self.b2]

    def forward(self, train_batch_x, train_batch_y):
        emb = self.embedding[train_batch_x]
        hidden = torch.tanh(emb.view(-1, self.embedding_size * self.block_size) @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        loss = torch.nn.functional.cross_entropy(logits, train_batch_y)
        return logits, loss

    def backward(self, loss):
        # Backward pass
        for par in self.params:
            par.grad = None

        loss.backward()
        for par in self.params:
            par.data -= self.learning_rate * par.grad

    def sample_from_distribution(self, idx_to_char):

        context = [0] * self.block_size
        embedded_context = self.embedding[context]
        word = []
        while True:
            linear_layer = embedded_context.view(-1, self.embedding_size * self.block_size) @ self.w1 + self.b1
            hidden = torch.tanh(linear_layer)
            logits = hidden @ self.w2 + self.b2
            probs = torch.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1, replacement=True).item()
            if idx == 0:
                break
            word.append(idx_to_char[idx])
            context = context[1:] + [idx]

        return word
