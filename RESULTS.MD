# Results

## Segmentation scores

*Results of Unigram Model of several settings of min size, max size segments, seeds and iterations. For evaluating we used 5 score functions. For more information see [docs](./README.md#score-functions). The model name in the table is described by the format of 'model min_size-max_size-iterations-init_segmentation (seed)'. Seed is by default 0. UM stands for Unigram Model, UMM for Unigram Model focused on each mode separately.*

|                      | **bacor accuracy (%)** | **bacor f1 (%)** | **mjww (%)** | **wtmf (%)** | **wufpc (pitches)** | **vocabulary size** | **average segment length** |
|----------------------|------------------------|------------------|--------------|--------------|---------------------|---------------------|----------------------------|
| UM 1-14-100-rand     | 67.80                  | 67.81            | 44.75        | **97.94**    | 3.77                | 50 103              | 12.62                      |
| UM 1-14-100-words    | 67.76                  | 67.74            | **45.33**    | 97.94        | **3.75**            | 50 193              | 12.62                      |
| UM 1-7-100-rand      | 91.66                  | 91.58            | 42.00        | 78.00        | 5.12                | 38 002              | 6.55                       |
| UM 1-7-100-words     | 91.73                  | 91.64            | 42.04        | 78.01        | 5.13                | 37 979              | 6.55                       |
| UM 3-8-100-rand      | 90.17                  | 90.04            | 41.54        | 85.66        | 4.90                | 47 668              | 7.50                       |
| UM 3-8-100-words     | 90.05                  | 89.95            | 41.54        | 85.60        | 4.90                | 47 709              | 7.50                       |
| UM 3-5-100-rand      | 93.34                  | **93.29**        | 36.57        | 59.78        | 5.87                | **13 495**          | 4.60                       |
| UM 3-5-100-words     | 93.17                  | 93.13            | 36.78        | 60.23        | 5.88                | 13 549              | **4.59**                   |
| _UMM 1-14-100-rand_  | _71.58_                | _72.55_          | _44.81_      | _99.59_      | _3.76_              | _50 601_            | _12.62_                    |
| _UMM 1-14-100-words_ | _71.53_                | _72.45_          | **_45.75_**  | **_99.61_**  | **_3.75_**          | _50 510_            | _12.62_                    |
| _UMM 1-7-100-rand_   | _95.89_                | _95.86_          | _41.25_      | _86.97_      | _5.12_              | _41 283_            | _6.56_                     |
| _UMM 1-7-100-words_  | _95.84_                | _95.82_          | _41.64_      | _87.07_      | _5.12_              | _41 192_            | _6.56_                     |
| _UMM 3-8-100-rand_   | _94.23_                | _94.20_          | _41.68_      | _92.61_      | _4.89_              | _50 465_            | _7.50_                     |
| _UMM 3-8-100-words_  | _94.16_                | _94.12_          | _41.80_      | _92.63_      | _4.89_              | _50 417_            | _7.50_                     |
| _UMM 3-5-100-rand_   | _97.69_                | _97.68_          | _36.52_      | _71.30_      | _5.90_              | **_15 711_**        | _4.60_                     |
| _UMM 3-5-100-words_  | **_97.91_**            | **_97.90_**      | _36.72_      | _71.44_      | _5.88_              | _15 735_            | **_4.59_**                 |


## Segmentation Statistics
### Unigram Model - Iteration Statistics
![accuracy](./models/imgs/unigram_model/unigram_model_accuracy.png)


![f1](./models/imgs/unigram_model/unigram_model_f1.png)


![mjww](./models/imgs/unigram_model/unigram_model_mjww.png)


![wtmf](./models/imgs/unigram_model/unigram_model_wtmf.png)


![wufpc](./models/imgs/unigram_model/unigram_model_wufpc.png)


![vocab](./models/imgs/unigram_model/unigram_model_vocab.png)


![avglen](./models/imgs/unigram_model/unigram_model_avglen.png)

### Unigram Model - All Modes are carried out separately - Iteration Statistics
![accuracy](./models/imgs/unigram_model_modes/unigram_model_modes_accuracy.png)


![f1](./models/imgs/unigram_model_modes/unigram_model_modes_f1.png)


![mjww](./models/imgs/unigram_model_modes/unigram_model_modes_mjww.png)


![wtmf](./models/imgs/unigram_model_modes/unigram_model_modes_wtmf.png)


![wufpc](./models/imgs/unigram_model_modes/unigram_model_modes_wufpc.png)


![vocab](./models/imgs/unigram_model_modes/unigram_model_modes_vocab.png)


![avglen](./models/imgs/unigram_model_modes/unigram_model_modes_avglen.png)



## Feature Extractions

### Frequencies of top 100 melodies

- Overlapping 4-grams
    ![top 4grams frequencies](./models/imgs/overlapping_ngrams/4grams_fe.png)
- Overlapping 5-grams
    ![top 5grams frequencies](./models/imgs/overlapping_ngrams/5grams_fe.png)
- Overlapping 6-grams
    ![top 6grams frequencies](./models/imgs/overlapping_ngrams/6grams_fe.png)
- Unigram Model (min size = 1, max size = 14, iterations = 100, init mode = gaussian)
    ![top UM frequencies 1 14 gaussian](./models/imgs/unigram_model/1_14_fe_random.png)
- Unigram Model (min size = 1, max size = 14, iterations = 100, init mode = words)
    ![top UM frequencies 1 14 words](./models/imgs/unigram_model/1_14_fe_words.png)
- Unigram Model (min size = 1, max size = 7, iterations = 100, init mode = gaussian)
    ![top UM frequencies 1 7 gaussian](./models/imgs/unigram_model/1_7_fe_random.png)
- Unigram Model (min size = 1, max size = 7, iterations = 100, init mode = words)
    ![top UM frequencies 1 7 words](./models/imgs/unigram_model/1_7_fe_words.png)
- Unigram Model (min size = 3, max size = 8, iterations = 100, init mode = gaussian)
    ![top UM frequencies 3 8 gaussian](./models/imgs/unigram_model/3_8_fe_random.png)
- Unigram Model (min size = 3, max size = 8, iterations = 100, init mode = words)
    ![top UM frequencies 3 8 words](./models/imgs/unigram_model/3_8_fe_words.png)
- Unigram Model (min size = 3, max size = 5, iterations = 100, init mode = gaussian)
    ![top UM frequencies 3 5 gaussian](./models/imgs/unigram_model/3_5_fe_random.png)
- Unigram Model (min size = 3, max size = 35, iterations = 100, init mode = words)
    ![top UM frequencies 3 5 words](./models/imgs/unigram_model/3_5_fe_words.png)
- Unigram Model Modes (min size = 1, max size = 14, iterations = 100, init mode = gaussian)
    ![top UM frequencies 1 14 gaussian](./models/imgs/unigram_model_modes/1_14_fe_random.png)
- Unigram Model Modes (min size = 1, max size = 14, iterations = 100, init mode = words)
    ![top UM frequencies 1 14 words](./models/imgs/unigram_model_modes/1_14_fe_words.png)
- Unigram Model Modes (min size = 1, max size = 7, iterations = 100, init mode = gaussian)
    ![top UM frequencies 1 7 gaussian](./models/imgs/unigram_model_modes/1_7_fe_random.png)
- Unigram Model Modes (min size = 1, max size = 7, iterations = 100, init mode = words)
    ![top UM frequencies 1 7 words](./models/imgs/unigram_model_modes/1_7_fe_words.png)
- Unigram Model Modes (min size = 3, max size = 8, iterations = 100, init mode = gaussian)
    ![top UM frequencies 3 8 gaussian](./models/imgs/unigram_model_modes/3_8_fe_random.png)
- Unigram Model Modes (min size = 3, max size = 8, iterations = 100, init mode = words)
    ![top UM frequencies 3 8 words](./models/imgs/unigram_model_modes/3_8_fe_words.png)
- Unigram Model Modes (min size = 3, max size = 5, iterations = 100, init mode = gaussian)
    ![top UM frequencies 3 5 gaussian](./models/imgs/unigram_model_modes/3_5_fe_random.png)
- Unigram Model Modes (min size = 3, max size = 35, iterations = 100, init mode = words)
    ![top UM frequencies 3 5 words](./models/imgs/unigram_model_modes/3_5_fe_words.png)