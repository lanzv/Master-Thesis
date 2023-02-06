# Master-Thesis
Unsupervised Melody Segmentation of Gregorian Chants

## Project Structure
 - data_analysis
   - cantus corpus dataset analysis
   - the melody_mode_frequencies contains images of melody mode frequencies of top 100 melodies extracted by feature extraction of SVC Linear
     - these images are also showed in the [RESULTS file](./RESULTS.MD)
 - research
   - There is a research [SUMMARIZATOIN](./research/ReadMe.MD) in the default readme file.
   - The [Observations](./research/Observations.MD) markdown file contains results and experiments of modified bacor project.
   - The related_works folder contains all papers related with this topic that could be used.
   - The theoretical_background folder contains all papers/sources describing all methods/models that are used in related works or just could be used in our case.
 - dataset
   - data generated by the bacor project
   - the aniphone_data.zip archive contains csv files of training and testing antiphone bacor's segmentations and cantus_corpus's data (modes, volpiano, etc...)
     - [loader.py](./models/src/utils/loader.py) uses these files
 - models
   - source code of the project
   - jupyter notebook experiments



## Experiments

Tabulated results of experiments could be seen in the [RESULTS file](./RESULTS.MD).
### Jupyter Notebooks experiments
 - [SEGMENTATION MODELS](./models/experiments.ipynb)
 - [Naive Bayes Experiments](./models/naive_bayes_analysis.ipynb)



## Models

### Overlapping n-grams
 - ***!Not A Correct Segmentation!*** - just for the analysis purpose
 - overlapping n grams of all chants parametrized by parameter n for n-gram
 - [source code](./models/src/models/overlapping_ngrams.py)

### Viterbi Based Model
 - gibbs sampling with init gaussian random solution
 - viterbi predicting new chant segmentation
 - parametrized min size and max size of segments, alpha of laplacian sampling, number of iterations, gaussian parameters, seed
 - [source code](./models/src/models/viterbi_based_model.py)





## Score functions

### bacor accuracy and f1
Segmentations are divided into training and testing dataset. The goal is to predict modes of chants the way the BACOR's paper does. Using the SVC Linear and TFIDF vectorizer we are trying to train and predict the correct modes of trainig datasets. Its accuracy and f1 are final scores. For the implementation details see the [source code](./models/src/eval/bacor_score.py).

### Melody Justified With Words (mjww)
Percentage of predicted segments that end with end of some word. The percentage is computed over all segments of all chants. For the implementation details see the [source code](./models/src/eval/mjww_score.py).

### Weighted Top Mode Frequency (wtmf)
For each melody, the mode frequency distribution is taken. The most common mode is considered and its frequency is used for the score evaluation. The final score value is evaluated as the average of all melodies. But each melody top mode frequency is counted as many times as the final sum of all melody occurencies. For the implementation details see the [source code](./models/src/eval/wtmf.py).

### Weighted Unique Final Pitch Count (wufpc)
The wufpc score is counted as average of all unique final pitches of all melodies in a single chant. But the averaging is weighted regarding the number of chant segments. So each computed value is counted as many times as the number of chant segments. For the implementation details see the [source code](./models/src/eval/wufpc_score.py).