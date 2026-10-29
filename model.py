from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from features import AlphabetDataset  # noqa: F401

RESOURCES = Path(__file__).parent / "res"


class Speech2Text(nn.Module):
    def __init__(
        self, nhead: int, num_layers: int, d_model: int = 128, num_classes: int = 26
    ) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(X.type(torch.float))[:, 0])


def train_s2t(dataset: Dataset, num_epochs: int, lr: float) -> Speech2Text:
    s2t = Speech2Text(2, 2)
    s2t.zero_grad()
    s2t.train()
    optimizer = optim.Adam(s2t.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    loss_func = nn.CrossEntropyLoss()

    progress = tqdm(total=num_epochs * len(loader))
    for _ in range(num_epochs):
        running_loss = torch.Tensor([0])

        for i, (xs, ys, _) in enumerate(loader):
            optimizer.zero_grad()
            logits = s2t.forward(xs)
            loss = loss_func.forward(logits, ys)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            average_loss = running_loss / (i + 1)
            progress.set_description(f"Loss (current epoch): {average_loss.item():.2f}")
            progress.update(1)

    s2t.eval()
    return s2t


def classify(s2t: Speech2Text, dataset: Dataset) -> torch.Tensor:
    probs = []
    loader = DataLoader(dataset, batch_size=64)
    for data, *_ in tqdm(loader):
        probs.append(F.softmax(s2t.forward(data), dim=-1))
    return torch.cat(probs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--classify",
        action="store_true",
        help="classify with pre-trained model (do not train)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=RESOURCES / "model.pt",
        help="model output file if training, or input file if classifying (.pt)",
    )
    parser.add_argument(
        "-d", "--data", default=RESOURCES / "data.pt", help="dataset (.pt) input path"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="number of epochs (if training)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="learning rate (if training)",
    )
    parser.add_argument(
        "-r",
        "--results",
        default=RESOURCES / "results.pt",
        help="results (.pt) output path",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if results_path.exists() and not results_path.is_file():
        raise OSError("Results path is not a file")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError("Dataset path does not exist")
    elif not data_path.is_file():
        raise OSError("Dataset path is not a file")

    print("Loading training data...")
    train = torch.load(data_path, weights_only=False)

    model_path = Path(args.model)
    if args.classify:
        if not model_path.exists():
            raise FileNotFoundError("Model path does not exist")
        elif not model_path.is_file():
            raise OSError("Model path is not a file")

        print("Loading model...")
        s2t = torch.load(model_path, weights_only=False)
    else:
        if model_path.exists() and not model_path.is_file():
            raise OSError("Model path is not a file")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        print("Training...")
        s2t = train_s2t(train, args.epochs, args.learning_rate)
        print("Saving model...")
        torch.save(s2t, RESOURCES / "model.pt")

    print("Classifying...")
    results = classify(s2t, train)
    print("Saving results...")
    torch.save(results, results_path)
    print("Done!")
