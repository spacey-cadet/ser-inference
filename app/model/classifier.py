import torch.nn as nn

IDX_TO_EMOTION = {
    0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
    4: "happy", 5: "neutral", 6: "sad", 7: "surprised",
}
NUM_CLASSES = 8


class MLPClassifier(nn.Module):
    """Must exactly match the architecture used at training time."""

    def __init__(self, input_size=1536, hidden=256, n_classes=8, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)
