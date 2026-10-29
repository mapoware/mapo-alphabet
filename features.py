from argparse import ArgumentParser
from pathlib import Path
from typing import Self

import numpy as np
import torch
from librosa.feature import melspectrogram
from numpy.typing import NDArray
from scipy.io import wavfile
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

RESOURCES = Path(__file__).parent / "res"


class AlphabetDataset(Dataset):
    def __init__(
        self, words: list[str], data: list[NDArray], labels: list[str]
    ) -> None:
        self.words = words
        self.data = data
        self.labels = labels

    def __getitem__(self, index) -> tuple[torch.Tensor, int, str]:
        return (
            torch.tensor(self.data[index]),
            ord(self.labels[index]) - ord("a"),
            self.words[index],
        )

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def from_folder(cls, folder: Path) -> Self:
        words = []
        labels = []
        specs = []

        print("Featurizing...")
        for path in tqdm(list(folder.glob("*.mp3"))):
            if not path.stem.isalpha():
                continue
            rate, data = wavfile.read(str(path))
            data = np.trim_zeros(data.astype(float))
            spec = melspectrogram(y=data, sr=rate)
            log_mel = np.log(np.where(spec == 0, np.finfo(float).eps, spec))

            words.append(path.stem)
            labels.append(path.stem[0])
            specs.append(log_mel.T)

        print("Padding data...")
        length = max(len(spec) for spec in specs)
        padded = [
            np.pad(spec, ((0, length - len(spec) + 1), (0, 0))) for spec in tqdm(specs)
        ]

        return cls(words, padded, labels)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Featurizes audio files with log-mel spectogram"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=RESOURCES / "audio",
        help="folder containing audio files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=RESOURCES / "data.pt",
        help="pytorch dataset (.pt) output path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError("Input path does not exist")
    elif not input_path.is_dir():
        raise OSError("Input path is not a directory")

    output_path = Path(args.output)
    if output_path.exists() and not output_path.is_file():
        raise OSError("Output path is not a file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = AlphabetDataset.from_folder(input_path)
    print("Saving...")
    torch.save(ds, output_path)
    print("Done!")
