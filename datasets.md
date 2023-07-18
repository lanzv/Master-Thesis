# Datasets

Our models are trained and evaluated primarily on the [Cantus Corpus](https://github.com/bacor/CantusCorpus) dataset. Furthermore, one of our score functions uses the [GregoBase Corpus](https://github.com/bacor/GregoBaseCorpus) dataset. These datasets provide a large amount of two genres of chants: antiphons and responsories. Antiphon melodies of Cantus Corpus include differentiae that are part of psalms corresponding to the antiphon. But they are not part of the antiphon itself. Therefore, we filter these differentiae from antiphon melodies. But we want to be able to directly compare our methods with proposals of the [Mode Classification and Natural Units in Plainchant](https://github.com/bacor/ISMIR2020) project that used the incorrect form of the antiphon dataset, so we keep all three datasets. The ```dataset/``` folder contains three zip files:
- ```antiphons.zip``` (filtered Cantus and GregoBase corpora corresponding to antiphons)
-  ```no4antiphons.zip``` (filtered Cantus and GregoBase corpora corresponding to antiphons without differentiae)
-  ```responsories.zip``` (filtered Cantus and GregoBase corpora corresponding to responsories)

Each zip file contains five files:
- ***gregobase-chantstrings.csv*** - filtered GregoBase chants of the particular genre using the Volpiano notation extended by the pause mark ```|``` indicating the end of melody phrase
- ***test-chants.csv*** - testing set of Cantus containing information about chants such as modes
- ***test-representation-pitch.csv*** - testing set of Cantus containing melody representation and proposed segmentations (words, n-grams, syllables, ...) by the [Mode Classification and Natural Units in Plainchant](https://github.com/bacor/ISMIR2020) project
- ***train-chants.csv*** - training set of Cantus containing information about chants such as modes
- ***train-representation-pitch.csv*** - training set of Cantus containing melody representation and proposed segmentations (words, n-grams, syllables, ...) by the [Mode Classification and Natural Units in Plainchant](https://github.com/bacor/ISMIR2020) project
  
Note that gregobase-chantstrings.csv is the same file for both antiphon and no4antiphon datasets since the GregoBase corpus does not keep differentiae in antiphon melodies.



## How to Use These Datasets for Our Experiments?

1. choose the dataset (antiphons, no4antiphons, responsories)
2. extract the corresponding zip file into the root directory of the repository
3. from the root folder, run the code
```python
from src.utils.loader import prepare_dataset, load_word_segmentations, load_syllable_segmentations, load_ngram_segmentations,load_phrase_segmentations


# Load list X of all dataset melodies (training+testing) represented as a string of tones, and the list of melody modes y
X, y = prepare_dataset(liquescents=False)

# Load word segmentation of all chants as a list of lists of string segments, where we keep liquescents
word_segmentation = load_word_segmentations(liquescents=True)

# Load syllable segmentation of all chants as a list of lists of string segments, where we keep liquescents
syllable_segmentation = load_syllable_segmentations(liquescents=True)

# Load 4-gram segmentation as a list of lists of string segments
ngram_segmentations = load_ngram_segmentations(n=4, liquescents=False)

# Load phrase segmentation of the GregoBase dataset
phrase_segmentation = load_phrase_segmentations(liquescents=False):
```

Our models could then use loaded melodies, modes, or segmentations.