# Results

## Segmentation scores

*Results of Unigram Model of several settings of min size, max size segments, seeds and iterations. For evaluating we used 5 score functions. For more information see [docs](./README.md#score-functions). The model name in the table is described by the format of 'model min_size-max_size-iterations-init_segmentation (seed)'. Seed is by default 0. UM stands for Unigram Model, UMM for Unigram Model focused on each mode separately.*

|                      | **bacor accuracy (%)** | **bacor f1 (%)** | **mjww (%)** | **wtmf (%)** | **wufpc (pitches)** | **vocabulary size** (last iteration - not final) | **average segment length** (last iteration - not final) |
|----------------------|------------------------|------------------|--------------|--------------|---------------------|---------------------|----------------------------|
| UM 1-14-100-rand     | 68.86                  | 69.03            | 44.86        | 97.75        | 3.77                | 50 103              | 12.62                      |
| UM 1-14-100-words    | 68.93                  | 68.92            | **45.47**    | **97.80**    | **3.75**            | 50 193              | 12.62                      |
| UM 1-7-100-rand      | 91.82                  | 91.74            | 42.24        | 77.99        | 5.12                | 38 002              | 6.55                       |
| UM 1-7-100-words     | 92.19                  | 92.10            | 42.27        | 78.03        | 5.11                | 37 979              | 6.55                       |
| UM 3-8-100-rand      | 90.17                  | 90.03            | 41.79        | 85.45        | 4.89                | 47 668              | 7.50                       |
| UM 3-8-100-words     | 90.12                  | 90.01            | 41.85        | 85.44        | 4.89                | 47 709              | 7.50                       |
| UM 3-5-100-rand      | **93.44**              | **93.39**        | 36.69        | 60.11        | 5.85                | **13 495**          | 4.60                       |
| UM 3-5-100-words     | 92.88                  | 92.84            | 36.96        | 60.42        | 5.87                | 13 549              | **4.59**                   |
| _UMM 1-14-100-rand_  | _73.53_                | _74.51_          | _44.90_      | _99.57_      | _3.76_              | _50 601_            | _12.62_                    |
| _UMM 1-14-100-words_ | _73.74_                | _74.73_          | **_45.89_**  | **_99.58_**  | **_3.75_**          | _50 510_            | _12.62_                    |
| _UMM 1-7-100-rand_   | _96.23_                | _96.20_          | _41.50_      | _87.27_      | _5.10_              | _41 283_            | _6.56_                     |
| _UMM 1-7-100-words_  | _96.08_                | _96.07_          | _41.86_      | _87.43_      | _5.11_              | _41 192_            | _6.56_                     |
| _UMM 3-8-100-rand_   | _94.57_                | _94.54_          | _41.96_      | _92.86_      | _4.88_              | _50 465_            | _7.50_                     |
| _UMM 3-8-100-words_  | _94.85_                | _94.83_          | _42.05_      | _92.90_      | _4.88_              | _50 417_            | _7.50_                     |
| _UMM 3-5-100-rand_   | _97.81_                | _97.81_          | _36.62_      | _71.82_      | _5.90_              | **_15 711_**        | _4.60_                     |
| _UMM 3-5-100-words_  | **_97.84_**            | **_97.83_**      | _36.86_      | _71.91_      | _5.87_              | _15 735_            | **_4.59_**                 |


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