# The MAPO Phonetic Alphabet
Code to reproduce the results used in the video [Minmaxing the NATO alphabet]() (TODO: insert link to final video).

## Requirements
```bash
pip install -r requirements
```

## Usage
First, generate audio files with
```bash
python tts.py
```
Then, featurize the audio files with
```bash
python features.py
```
Finally, train the model and classify the data with
```bash
python model.py
```

The results can then be analyzed to produce the final alphabets with `analysis.ipynb` (alphabets may not match video due to randomness during training).
