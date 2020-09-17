# Translation sequence to sequence

## About

This an excersice and implementation of an attention RNN to create a French to English translator.
It uses a data set that consist of over 135000 pairs of sentences, where one pair consist of a sentence in English with its corresponding translation to French.

For this excersice only sentences consisting of 10 words or less were used.

## Prerequisites

The notebook was run using GPU.
I use anaconda and therefore I use conda to create my virtual environments. The following packages are needed to run the notebook

```
__future__division
io
unicodedata
string
re
random
time
torch
jupyter
```

## Predictions

Here some of the translation the model was able to do after its training:

```
French sentence:     il lit un roman en ce moment .
Correct translation: he s reading a novel right now .
Model translated:    hesreadinganovelrightnow.

French sentence:     je vais avoir trente ans en octobre .
Correct translation: i m turning thirty in october .
Model translated:    imthirtyallthisinlove.

French sentence:     nous sommes anxieux .
Correct translation: we re anxious .
Model translated:    weresurprised.
```
