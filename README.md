# The MAPO Phonetic Alphabet
Code to reproduce the results used in the video [Minmaxing the NATO alphabet](https://www.youtube.com/watch?v=By13Xxnaz-Q).

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
(each of the scripts have additional arguments for file paths, but they shouldn't be necessary for most use cases)


The results can then be analyzed to produce the final alphabets with `analysis.ipynb` (alphabets may not match video due to randomness during training).
