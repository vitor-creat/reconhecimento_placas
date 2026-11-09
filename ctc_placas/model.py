import torch
import torch.nn as nn

class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden*2, out)
    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return out


class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            self._ConvLayer(in_channels, 64, kernel=3, stride=1, padding=1, bn=True),
            nn.MaxPool2d(2, 2),

            self._ConvLayer(64, 128, kernel=3, stride=1, padding=1, bn=True),
            nn.MaxPool2d(2, 2),

            self._ConvLayer(128, 256, kernel=3, stride=1, padding=1, bn=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            self._ConvLayer(256, 256, kernel=3, stride=1, padding=1, bn=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.AdaptiveAvgPool2d((1, None))
        )

        self.linear = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output+1)

    def forward(self, X, y=None, criterion = None):
        out = self.cnn(X)
        N, C, w, h = out.size()

        out = out.view(N, -1, h)
        out = out.permute(0, 2, 1)
        out = self.linear(out)

        out = out.permute(1, 0, 2)
        out = self.rnn(out)

        if y is not None:
            T = out.size(0)
            N = out.size(1)

            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=7, dtype=torch.int32)

            loss = criterion(out.log_softmax(2), y, input_lengths, target_lengths)

            return out, loss

        return out, None

    def _ConvLayer(self, inp, out, kernel, stride, padding, bn=False):
        if bn:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(out)
            ]
        else:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU()
            ]
        return nn.Sequential(*conv)