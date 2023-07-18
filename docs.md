# Documentation

We propose several unsupervised segmentation models and evaluation functions that analyze the predicted segmentation. All is implemented in Python (eventually Cython) and placed in the ```src/``` directory. The directory is structured this way:
- ```src/```
  - ```eval/```
    - evaluating segmentation on our score functions
    - evaluation pipelines
    - script to do the (top 100) feature extraction
  - ```models/```
    - unigram models
    - NHPYLM models
    - BERT
    - random baseline
    - overlapping n-grams upper bound
  - ```utils/```
    - plotters
    - data loaders ([datasets.md](datasets.md))
    - training iteration statistics
    - dataset analysis
    - gregobase preparation


## Models - usage

First of all, install all necessary packages:
```sh
pip install -r requirements.txt
```

### UM (unigram model)

Suppose we have list of string melodies ```X_train```, ```X_dev```, and ```X_test``` and their corresponding modes ```y_train```, ```y_dev```. Then we can get the test segmentation, perplexity, and mawp score this way:
```Python
from src.models.unigram_model import UnigramModel

# Init model
model = UnigramModel(3, 5)
# Train model
model.train(X_train, X_dev, y_train, y_dev, iterations=100, init_mode = "gaussian", print_each=10)
# Predictions
mawp_score = model.get_mawp_score()
test_segmentation, test_perplexity = model.predict_segments(X_test)
```


### UMM (unigram model modes)

Suppose we have list of string melodies ```X_train```, ```X_dev```, and ```X_test``` and their corresponding modes ```y_train```, ```y_dev```. Then we can get the test segmentation, perplexity, and mawp score this way:
```Python
from src.models.unigram_model import UnigramModelModes

# Init model
model = UnigramModelModes(3, 5)
# Train model
model.train(X_train, X_dev, y_train, y_dev, iterations=100, init_mode = "gaussian", print_each=10)
# Predictions
mawp_score = model.get_mawp_score()
test_segmentation, test_perplexity = model.predict_segments(X_test)
```


### NHPYLM
Because of the time complexity, we implemented the Nested Hierarchical Pitman-Yor Language Model in Cython, which made the training significantly faster. The implementation of the NHPYLM model is placed in the ```src/models/nhpylm``` folder called by the NHPYLMModel class implemented in ```src/models/nhpylm_model.pyx```. To use the model, we need to set up the Cython code first.

```sh
python setup.py build_ext --inplace
```

Then we can initialize and train the NHPYLM model from Python code.

```Python
from nhpylm_model import NHPYLMModel

# Init model
model = NHPYLMModel(7, init_d = 0.5, init_theta = 2.0,
                init_a = 6.0, init_b = 0.83333333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0)
# Train model
model.train(X_train, X_dev, y_train, y_dev, 200, True, True, print_each_nth_iteration=10)
# Predictions
mawp_score = model.get_mawp_score()
test_segmentation, test_perplexity = model.predict_segments(X_test)
```


### NHPYLMModes
NHPYLMModes is the extension of the NHPYLM model based on eight separate NHPYLM submodels for each chant mode. The NHPYLMModes class is also implemented in Cython in the ```src/models/nhpylm_model.pyx``` file. Therefore, as in the previous case, if we want to use this model, we have to set up the Cython code first.

```sh
python setup.py build_ext --inplace
```

The model could be used in Python this way:

```Python
from nhpylm_model import NHPYLMModesModel

# Init model
model = NHPYLMModesModel(7, init_d = 0.5, init_theta = 2.0,
                init_a = 6.0, init_b = 0.83333333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0)
# Train model
model.train(X_train, X_dev, y_train, y_dev, 200, True, True, print_each_nth_iteration=50)
# Predictions
mawp_score = model.get_mawp_score()
test_segmentation, test_perplexity = model.predict_segments(X_test)
```

Note that the NHPYLMModes model needs eight times more time for segment predictions. Be careful with the number of printed iterations.

### BERT

We use the official implementation of the [Unsupervised Chinese Word Segmentation with BERT Oriented Probing and Transformation](https://github.com/QbethQ/Unsupervised_CWS_BOPT) paper. We only adapted the code to our task, and we extended the model so that we could pretrain it first. Setting files of our task's BERT model are placed in the ```bert/``` folder (**config.json**, **vocab.txt**).

For the pretraining process, install the packages with their particular version.
```sh
pip install transformers==4.27.4 tokenizers==0.13.2
```

Then, move the **bert/config.json** and **bert/vocab.txt** into the root directory. Choose the dataset (**bert/antiphons.zip**, **bert/no4antiphons.zip**, **bert/responsories.zip**), and extract it in the root directory. Then you can pretrain BERT using:

``` Python
from src.models.bert_model import BERT_Model

model = BERT_Model(working_dir=".")
model.pretrain()
```

The new folder ```PretrainedBERT_chant/``` should be created. For the training process, change back the version of transformers:

```sh
pip install transformers==2.8.0
```

And the model could be trained again in the Python code.

``` Python
from src.models.bert_model import BERT_Model

model = BERT_Model(working_dir=".")
model.train()
```

For the prediction, the dataset format stored in the ```datasets/``` folder (not **bert/antiphons.zip**, **bert/no4antiphons.zip**, **bert/responsories.zip**) are used. 

``` Python
from src.models.bert_model import BERT_Model

# Init model
model = BERT_Model(working_dir=".")

# Predictions
mawp_score = model.get_mawp_score()
test_segmentation, _ = model.predict_segments(X_test)
```

### Random baseline

The RandomModel generates random segmentation.
```Python
from src.models.random_model import RandomModel

# Init model
model = RandomModel(1, 7)
# Predictions
mawp_score = model.get_mawp_score()
test_segmentation, test_perplexity = model.predict_segments(X_test)
```

### Overlapping N-grams upper bound
The Overlapping N-grams is not a segmentation since all segments overlap with others. Furthermore, we can combine more Overlapping N-grams. However, we can measure only the mode classification score. The code sample using the OverlappingNgram class is shown below. 

```Python
from src.utils.loader import prepare_dataset
from src.models.overlapping_ngrams import OverlappigNGrams
from src.eval.pipelines import bacor_pipeline

# Get Data
X, y = prepare_dataset()
# Init models
model1 = OverlappigNGrams(1)
model2 = OverlappigNGrams(2)
model3 = OverlappigNGrams(3)
model4 = OverlappigNGrams(4)
model5 = OverlappigNGrams(5)
model6 = OverlappigNGrams(6)
model7 = OverlappigNGrams(7)
# Train and Fit model
final_segmentation = []
final_segmentation1 = model1.predict_segments(X)
final_segmentation2 = model2.predict_segments(X)
final_segmentation3 = model3.predict_segments(X)
final_segmentation4 = model4.predict_segments(X)
final_segmentation5 = model5.predict_segments(X)
final_segmentation6 = model6.predict_segments(X)
final_segmentation7 = model7.predict_segments(X)
for c1, c2, c3, c4, c5, c6, c7 in zip(final_segmentation1,final_segmentation2,
                                      final_segmentation3,final_segmentation4,
                                      final_segmentation5,final_segmentation6,
                                      final_segmentation7):
  final_segmentation.append(c1+c2+c3+c4+c5+c6+c7)
# Evaluate model
scores, selected_features, trained_model = bacor_pipeline(
    final_segmentation, y, max_features_from_model = 100, include_additative = False, fe_occurence_coef = 10, all_features_vectorizer=True)
```


## Evaluation scores - usage
Once we have the segmentation (of both training and testing datasets), mawp function, and perplexities, we can evaluate all score functions and visualizations we proposed by a single command.

```Python
from src.eval.pipelines import evaluation_pipeline

bacor_model = evaluation_pipeline(
    train_segmentation, y_train, test_segmentation, y_test, train_perplexity, test_perplexity, mawp_score,
    max_features_from_model = 100, include_additative = False, fe_occurence_coef=10)
```

The command will first evaluate and print score results and charts of the training dataset. Then the testing dataset will be evaluated. At the end of the pipeline, the top 100 feature extraction is done and printed with all its related charts.

We measure these score functions
- **perplexity**: the ability of the probability model to predict a sample.
- melody segmentation scores
  - **bacor_accuracy**: mode classification accuracy score of segmentation using the TFIDF vectorizer and SVC classifier
  - **bacor_f1**: mode classification f1 score of segmentation using the TFIDF vectorizer and SVC classifier
  - **nb_accuracy**: mode classification accuracy score of segmentation using the TFIDF vectorizer and Naive Bayes classifier
  - **nb_f1**: mode classification f1 score of segmentation using the TFIDF vectorizer and Naive Bayes classifier
  - **avg_seg_len**: average segment length of all segments in the testing dataset
  - **vocab_size**: the size of segment vocabulary considering the testing dataset - number of unique melodic units used in the segmentation 
  - **maww**: how well is melody aligned with words
  - **mawp**: how well is melody aligned with phrases
  - **wtmf**: weighted top mode frequency, weighted (by its frequency) average of percentages that the melodic unit is used in its dominant mode considering all segment occurrences
  - **vocab_levenshtein**: score measuring the diversity of the vocabulary, the average of medians of levensthein distances between segments 
  - **wufpc**: weighted unique final pitch count, number of unique last tones of melody segments
- charts
  - **unique segment density**: charts that visualize the percentage of unique segments at the specific position over all segments at that position
  - **average segment length**: charts that visualize average segment length at the specific position of the chant
  - **segment occurrences**: charts that visualize segment occurrences of all vocabulary melodic units and how much modes share the same segments