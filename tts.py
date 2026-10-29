from argparse import ArgumentParser
from pathlib import Path

import pyttsx3

RESOURCES = Path(__file__).parent / "res"


def save_tts_files(words: list[str], output_path: Path) -> None:
    engine = pyttsx3.init()
    for word in words:
        engine.save_to_file(word, str((output_path / f"{word}.mp3").absolute()))
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    parser = ArgumentParser("Generates audio files from word list")
    parser.add_argument(
        "-i", "--input", default="words.txt", help="file with list of words"
    )
    parser.add_argument(
        "-o", "--output", default=RESOURCES / "audio", help="output directory"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError("Input path does not exist")
    elif not input_path.is_file():
        raise OSError("Input path is not a file")

    output_path = Path(args.output)
    if output_path.exists() and not output_path.is_dir():
        raise OSError("Output path is not a directory")
    output_path.mkdir(parents=True, exist_ok=True)

    words = []
    with input_path.open() as file:
        for line in file.readlines():
            words.append(line.strip())

    print("Generating audio files...")
    save_tts_files(words, output_path)
    print("Done!")
