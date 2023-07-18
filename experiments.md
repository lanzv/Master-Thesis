# Experiments

We propose several unsupervised segmentation methods and several evaluation metrics and visualizations ([docs.md](docs.md)). We also consider some baselines and upper bounds to have something we can compare our models with. We measure some additional experiments to support our hypotheses. But also we analyze both datasets (Cantus and GregoBase) and their properties. The outcomes of these experiments and analyses are stored in the ```notebooks/``` folder as outputs of jupyter notebook's cells. Furthermore, we provide the best practices for using our models and evaluation functions. 

**Note that the in the case we would want to run cells of the particular jupyter notebook, we have to place the notebook into the root directory (with the extracted dataset files as described in [datasets.md](datasets.md)). The jupyter notebook needs to be in the same directory as the ```src/``` folder**

## Notebooks

Each experiment type of a particular dataset (antiphons / no4antiphons / responsories) is measured in one jupyter notebook file. 

- ***antiphons.ipynb***: the notebook contains all models we propose and their results of all evaluation metrics on the antiphon dataset
- ```antiphons/```
  - ***baselines.ipynb***: the notebook evaluates *Words* segmentation proposed by [Cornelissen](https://github.com/bacor/ISMIR2020) and the  *Rand* segmentation on the antiphon dataset 
  - ***NHPYLMModes_5seeds.ipynb***: the notebook checks the validness of the *NHPYLMModes* model by shuffling gold data of modes on the antiphon dataset
  - ***overlapping_n_grams.ipynb***: the notebook evaluates the *NgramOverlap* to get the upper bound on the antiphon dataset
  - ***trimmed_experiments.ipynb***: the notebook measures the experiment of removing segments from the left, right, or both sides at the same time of chant melodies on the antiphon dataset 
- ***no4antiphons.ipynb***: the notebook contains all models we propose and their results of all evaluation metrics on the antiphons-without-differentiae dataset
- ```no4antiphons/```
  - ***baselines.ipynb***: the notebook evaluates *Words* and *ngram* segmentations proposed by [Cornelissen](https://github.com/bacor/ISMIR2020) and the  *Rand* segmentation on the antiphons-without-differentiae dataset 
  - ***NHPYLMModes_5seeds.ipynb***: the notebook checks the validness of the *NHPYLMModes* model by shuffling gold data of modes on the antiphons-without-differentiae dataset 
  - ***overlapping_n_grams.ipynb***: the notebook evaluates the *NgramOverlap* to get the upper bound on the antiphons-without-differentiae dataset
  - ***trimmed_experiments.ipynb***: the notebook measures the experiment of removing segments from the left, right, or both sides at the same time of chant melodies on the antiphons-without-differentiae dataset
- ***responsories.ipynb***: the notebook contains all models we propose and their results of all evaluation metrics on the responsory dataset
- ```responsories/```
  - ***baselines.ipynb***: the notebook evaluates *Syllables* segmentation proposed by [Cornelissen](https://github.com/bacor/ISMIR2020) and the *Rand* segmentation on the responsory dataset 
  - ***NHPYLMModes_5seeds.ipynb***: the notebook checks the validness of the *NHPYLMModes* model by shuffling gold data of modes on the responsory dataset 
  - ***overlapping_n_grams.ipynb***: the notebook evaluates the *NgramOverlap* to get the upper bound on the responsory dataset
  - ***trimmed_experiments.ipynb***: the notebook measures the experiment of removing segments from the left, right, or both sides at the same time of chant melodies on the responsory dataset
- ***corpus_analysis.ipynb***: the notebook analyzes both corpora (Cantus and GregoBase) and their properties
- ***phrases_dataset.ipynb***: the notebook provides the guideline on how to prepare the filtered GregoBase dataset containing pause marks ```|``` based on the GregoBase corpus

## Results

As part of this section, we list the baseline results compared with our proposals.

### Antiphons
| |bacor_accuracy | bacor_f1 | nb_accuracy | nb_f1 |
|---|---|---|---|---|
| Rand | 87.59 | 87.39 | 81.37 | 82.20 |
| Words_liq | 94.76 | 94.71 | 90.17 | 90.30 |
| Words | 95.22 | 95.18 | 91.01 | 91.10 |
| UM3_5 | 93.58 | 93.53 | 88.15 | 88.43 |
| UM1_7 | 94.52 | 94.47 | 88.70 | 89.06 |
| UMM3_5 | 93.99 | 93.98 | 93.03 | 93.03 |
| UMM1_7 | 92.69 | 92.64 | 83.34 | 83.01 |
| NHPYLM | 95.77 | 95.75 | 93.94 | 94.03 |
| NHPYLMModes | **96.03** | **96.03** | **96.18** | **96.18** |
| BERT | 89.52 | 89.39 | 81.80 | 82.26 |
| NgramOverlap | *96.13* | *96.11* | *92.74* | *92.59* |

| 	 | perply | vocab_size | avg_seg_len | vocab_levenshtein |
|---|---|---|---|---|
| Rand | - | 14986 | 4.28 | 0.91 |
| Words_liq | - | 11087 | 3.77 | 0.91 |
| Words | - | 9434 | 3.77 | 0.90 |
| UM3_5 | 818.79 | 2325 | 4.26 | 0.99 |
| UM1_7 | 1517.45 | 3581 | 4.96 | 0.93 |
| UMM3_5 | 441.70 | 3143 | 4.06 | 0.97 |
| UMM1_7 | 367.89 | 3782 | 3.85 | 0.92 |
| NHPYLM | 25.20 | 2161 | 2.44 | 0.90 |
| NHPYLMModes | 28.10 | 3493 | 2.78 | 0.93 |
| BERT | - | 6300 | 1.85 | 0.82 |

| 	 | wtmf | maww | mawp | wufpc |
|---|---|---|---|---|
| Rand | 64.61 | 26.89 | 37.20 | 5.82 |
| Words_liq | 63.00 | 100.00 | - | 6.23 |
| Words | 61.71 | 100.00 | - | 5.73 |
| UM3_5 | 54.91 | 32.76 | 46.83 | 5.59 |
| UM1_7 | 57.92 | 33.10 | 51.63 | 5.33 |
| UMM3_5 | 64.21 | 34.45 | 48.23 | 5.71 |
| UMM1_7 | 59.90 | 38.66 | 56.49 | 5.85 |
| NHPYLM | 49.27 | 47.87 | 71.81 | 7.73 |
| NHPYLMModes | 60.50 | 49.32 | 67.23 | 6.27 |
| BERT | 48.09 | 57.75 | 64.38 | 6.92 |


### Antiphons without differentiae
| 	 | bacor_accuracy | bacor_f1 | nb_accuracy | nb_f1 |
|---|---|---|---|---|
| Rand | 81.70 | 81.08 | 75.33 | 76.68 |
| Words_liq | 90.28 | 90.16 | 86.57 | 86.84 |
| Words | 90.53 | 90.38 | 86.72 | 86.98 |
| 4gram_liq | 91.27 | 91.14 | 83.25 | 83.62 |
| UM3_5 | 90.11 | 90.00 | 84.50 | 84.94 |
| UM1_7 | 89.84 | 89.69 | 84.99 | 85.50 |
| UMM3_5 | 89.82 | 89.77 | 87.43 | 87.40 |
| UMM1_7 | 88.12 | 87.91 | 76.21 | 75.92 |
| NHPYLM | 92.99 | 92.90 | 91.07 | 91.31 |
| NHPYLMModes | **94.02** | **94.01** | **93.58** | **93.59** |
| BERT | 87.28 | 87.11 | 79.46 | 80.02 |
| NgramOverlap | *94.69* |*94.65* |*90.14* | *89.95* |


| 	 | perplexity | vocab_size | avg_seg_len | vocab_levenshtein |
|---|---|---|---|---|
| Rand | - | 14281 | 4.26 | 0.91 |
| Words_liq | - | 10780 | 3.63 | 0.91 |
| Words | - | 9201 | 3.63 | 0.90 |
| 4gram_liq | - | 4211 | 3.89 | 1.00 |
| UM3_5 | 825.84 | 2270 | 4.25 | 0.99 |
| UM1_7 | 1511.34 | 3407 | 4.83 | 0.93 |
| UMM3_5 | 525.07 | 3122 | 4.01 | 0.98 |
| UMM1_7 | 401.23 | 3731 | 3.58 | 0.92 |
| NHPYLM | 26.10 | 2353 | 2.34 | 0.90 |
| NHPYLMModes | 31.08 | 3317 | 2.69 | 0.93 |
| BERT | - | 6654 | 1.97 | 0.84 |


| 	 | wtmf | maww | mawp | wufpc |
|---|---|---|---|---|
| Rand | 63.39 | 27.37 | 37.44 | 5.79 |
| Words_liq | 61.50 | 100.00 | - | 6.22 |
| Words | 60.09 | 100.00 | - | 5.72 |
| 4gram_liq | 55.05 | 29.51 | - | 6.36 |
| UM3_5 | 52.79 | 32.39 | 48.79 | 5.58 |
| UM1_7 | 52.44 | 31.07 | 52.44 | 5.38 |
| UMM3_5 | 60.75 | 34.29 | 51.26 | 5.61 |
| UMM1_7 | 54.91 | 37.74 | 58.13 | 6.00 |
| NHPYLM | 49.63 | 46.78 | 71.59 | 7.72 |
| NHPYLMModes | 55.99 | 47.78 | 67.91 | 6.28 |
| BERT | 47.71 | 52.74 | 62.84 | 7.19 |

### Responsories
|	 |bacor_accuracy |bacor_f1 |nb_accuracy | nb_f1 |
|---|---|---|---|---|
| Rand | 82.12 | 81.93 | 75.11 | 76.09 |
| Syllables_liq | 92.70 | 92.68 | 89.81 | 89.95 |
| Syllables | 93.27 | 93.25 | 89.43 | 89.55 |
| UM3_5 | 92.18 | 92.13 | 84.73 | 84.89 |
| UM1_7 | 92.41 | 92.38 | 86.39 | 86.59 |
| UMM3_5 | 91.18 | 91.18 | 90.61 | 90.62 |
| UMM1_7 | 89.47 | 89.45 | 79.66 | 78.94 |
| NHPYLM | 93.12 | 93.12 | 91.13 | 91.23 |
| NHPYLMModes | **94.22** | **94.22** | **94.22**| **94.21** |
| BERT | 87.43 | 87.37 | 75.91 | 76.49 |
| NgramOverlap | *94.31* | *94.30* | *93.22* | *93.20* |
| 6gramOverlap | *95.21* | *95.20* | *91.99* | *91.92* |

|	 |perplexity |vocab_size |avg_seg_len |vocab_levenshtein |
|---|---|---|---|---|
| Rand | - | 16839 | 4.37 | 0.91 |
| Syllables_liq | - | 7342 | 2.92 | 0.90 |
| Syllables | - | 6907 | 2.92 | 0.90 |
| UM3_5 | 978.36 | 2625 | 4.44 | 0.99 |
| UM1_7 | 1972.99 | 4443 | 5.22 | 0.94 |
| UMM3_5 | 523.54 | 3336 | 4.16 | 0.98 |
| UMM1_7 | 475.59 | 4447 | 4.11 | 0.94 |
| NHPYLM | 22.92 | 2676 | 2.68 | 0.92 |
| NHPYLMModes | 24.99 | 4170 | 2.93 | 0.93 |
| BERT | - | 4862 | 1.42 | 0.76 |

|	 |wtmf |maww |mawp |wufpc |
|---|---|---|---|---|
| Rand | 57.59 | 26.22 | 27.20 | 7.11 |
| Syllables_liq | 49.75 | 100.00 | - | 9.22 |
| Syllables | 49.37 | 100.00 | - | 7.63 |
| UM3_5 | 47.49 | 35.31 | 39.37 | 7.03 |
| UM1_7 | 52.94 | 36.29 | 44.60 | 6.89 |
| UMM3_5 | 56.93 | 38.23 | 43.27 | 7.05 |
| UMM1_7 | 56.29 | 41.30 | 54.67 | 7.23 |
| NHPYLM | 46.15 | 55.49 | 76.52 | 8.84 |
| NHPYLMModes | 53.96 | 54.06 | 68.80 | 7.53 |
| BERT | 41.04 | 69.23 | 81.04 | 8.43 |


## Top 100 features

Using our feature extraction method (taking 100 most frequent segments from the top 1000 features based on the SVC coefficients), we get 100 features of both antiphons and responsories of segmentations generated by the *NHPYLMModes* model.

### Antiphons (without differentiae)
|  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| g | k | h | f | d | l | hg | e | gh | gg |
| fe | fed | gf | efg | j | ff | hh | fg | df | c |
| ghg | kk | kj | lk | ll | i | fgh | cd | m | hk |
| dc | hgfg | ed | kjh | ghgf | cdd | fgg | dd | fh | fghg |
| hgh | fd | efgfedd | hhg | hjhgg | dcd | fghhgg | ggg | jk | kkjh |
| ee | fedd | jkl | fghhg | kjhg | hgg | cdf | kkj | hgf | hj |
| lml | hkh | de | fghh | lm | fgf | jh | fefg | ddcfg | kkl |
| hghg | hgfgg | cdfedd | gfed | ccd | ghgg | hjhg | ghk | hhgg | efgfed |
| fdc | defg | lkj | gfg | cddd | ki | jklk | fef | eg | dee |
| kh | fedcd | jkhg | fhk | ffg | efgg | klk | hkhg | ghgfg | lll |


### Responsories

|  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| f | g | h | k | d | e | l | gh | j | hjkjhg |
| hg | gf | kj | c | fe | kl | gg | hjhghhg | hk | ghg |
| kk | fg | hhg | cd | i | lm | hh | fed | efg | df |
| dd | hkhghhg | ed | fgh | dc | jklkj | defefed | ff | kkj | ll |
| jk | fd | hgh | fh | nm | jh | fghg | ghgfggf | ffe | fgf |
| m | jkl | ghgg | klk | cdf | hgfg | hjh | dcd | gfgh | hkghg |
| ef | efd | cdd | hghg | eed | hgfghg | defed | hgf | ghkj | hghgfgg |
| efgfggf | gfg | efed | ln | fgfe | hjkjhgg | hih | gghg | fdf | lk |
| ghhg | llk | hkh | hgg | efedefd | fghh | egfffe | fhk | lkk | efede |
| fghghhg | dfd | hkk | ggg | defedcd | kjhghg | ggf | gfed | gff | klkj |

## Conclusion

Based on the results from the notebooks, we can conclude:

 1. **Natural segmentation by words or syllables is not ideal.** 
 2. **The beginnings and ends of chants have a stronger modal identity than the segments in the middle.**
 3. **Many segments are shared among all modes, but there are also segments that are used by only one mode.**
 4. **Conditional NHPYLM model generates the segmentation that gives the state-of-the-art performance on the mode classification task based on the melody segmentation.**
 5. **There are only a few frequent segments and a lot of occasional ones.**