# Results

## Segmentation scores

*Results of Unigram Model of several settings of min size, max size segments, seeds and iterations. For evaluating we used 5 score functions. For more information see [docs](./README.md#score-functions). The model name in the table is described by the format of 'VBM min_size-max_size-iterations (seed)'. Seed is by default 0.*

|                  | **bacor accuracy (%)** | **bacor f1 (%)** | **mjww (%)** | **wtmf (%)** | **wufpc (pitches)** |
|------------------|------------------------|------------------|--------------|--------------|---------------------|
| UM 4-6-20        | 92.69                  | 92.62            | 39.75        | 69.16        | 5.40                |
| UM 4-5-20        | 92.81                  | 92.75            | 36.20        | 60.55        | 5.97                |
| UM 3-5-1 (0)     | 92.76                  | 92.71            | 35.96        | 59.39        | 5.95                |
| UM 3-5-5 (0)     | 93.17                  | 93.13            | 36.38        | 60.00        | 5.86                |
| UM 3-5-10 (0)    | 93.24                  | 93.20            | 36.46        | 59.98        | 5.83                |
| UM 3-5-20 (0)    | 93.20                  | 92.96            | 36.61        | 60.02        | 5.84                |
| UM 3-5-1 (153)   | 92.67                  | 92.60            | 35.97        | 59.42        | 5.93                |
| UM 3-5-5 (153)   | 92.91                  | 92.87            | 36.45        | 60.00        | 5.84                |
| UM 3-5-10 (153)  | 93.24                  | 93.19            | 36.39        | 59.87        | 5.85                |
| UM 3-5-20 (153)  | **93.41**              | **93.36**        | 36.41        | 59.86        | 5.86                |
| UM 1-7-20        | 91.70                  | 91.61            | **41.95**    | 77.72        | 5.11                |
| UM 2-8-20        | 89.66                  | 89.51            | 41.11        | **85.54**    | **4.89**            |


## Feature Extractions

### Frequencies of top 100 melodies

- Overlapping 4-grams
    ![top 4grams frequencies](./data_analysis/melody_mode_frequencies/4grams.png)
- Overlapping 5-grams
    ![top 5grams frequencies](./data_analysis/melody_mode_frequencies/5grams.png)
- Overlapping 6-grams
    ![top 6grams frequencies](./data_analysis/melody_mode_frequencies/6grams.png)
- Unigram Model (min size = 2, max size = 8, iterations = 20)
    ![top VBM frequencies](./data_analysis/melody_mode_frequencies/vbm.png)