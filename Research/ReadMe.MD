# Research of Unsupervised Melody Segmentation of Gregorian Chants

- Questions
  - ***Unsupervised Model to use for training***
  - ***Evaluation methods***
  - ***Loss function in case of Neural Networks***

- Goal
  - ***Defeat [this](./related_works/ISMIR2020.pdf) article of Gregorian Chant Mode recognition***

## Theoretical Background
### Language Models and Probability Methods
1. [2019 - BERT](./theoretical_background/2019_BERT.pdf)
    - Bidirectional Encoder Representation from Transformers
    - Encoder (Positional Encoding, Feed Forward, Multi-Head Attention and some Normalisation)
    - Decoder (Positional Encoding, Feed Forward, Multi-Head Attention, Masked Multi-hEad Attention)
    - Softmax from Decoder, output goes to decoder input of the next output
    - Two phases
      - Pretraining - Masked Language Modeling, Next Sentence Prediction, sum of Token, Segment and Position Embeddings
      - Fine Tuning - Specific Task learning (another kind of output)
    - [YouTube Summarization](https://www.youtube.com/watch?v=xI0HHN5XKDo&ab_channel=CodeEmporium)
1. [2019 - RoBERTa](./theoretical_background/2019_RoBERTa.pdf)
    - Robustly optimized BERT approach
    - trained with dynamic masking
    - FULL-SENTENCES without NSP loss
    - large mini-batches
    - larger byte-leve Byte-Pair Encoding (BPE)
    - [YouTube Summarization](https://www.youtube.com/watch?v=-MCYbmU9kfg&ab_channel=YannicKilcher)
1. [Pitman-Yor](theoretical_background/1996_The_two-parameter_Poisson-Dirichlet_distribution_derived_from_a_stable_subordinator.pdf)
    - Based on [Chinese Restaurant Process (YouTube Intuition)](https://www.youtube.com/watch?v=4wa2WyDrgMA&ab_channel=JordanBoyd-Graber)
    - Language model that consider n-grams in a pyramide scheme stored as tables in multiple restaurants and the "unigram restaurant"
    - Final probability is computed as a sum of existing table probability and new table probability
    - Pitman-Yor assumes duplicate tables of CRP
    - Kneser-Ney assumes one table with a dish (word) per restaurant
    - [Gibbs Sampling](https://www.youtube.com/watch?v=7LB1VHp4tLE&ab_channel=ritvikmath) of the seating assignments is required (kind of Markov Chain Monte Carlo algorithm -> gives us sequence of observations)
1. [Naive Baise](./theoretical_background/2006_Naive_Bayes_classifiers.pdf)
    - Naive Baise Classifier
    - Get counts of classes from dataset ~ P(cl1)...
    - Compute probability distribution of features, in case of Gaussian NB compute Gaussian curve with its Mean and Standard Deviations
    - multiply all probabilities together (or in case of using logs sum it)
    - The class with highest probability wins
    - [YouTube NB Summarization](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
    - [YouTube Gaussian NB Summarization](https://www.youtube.com/watch?v=H3EjCKtlVog&ab_channel=StatQuestwithJoshStarmer)
  
### Gregorian Chants
1. [Gregorian Chants](./related_works/HILEY__David_Gregorian_Chant.pdf)


## Inspirations from other topics

### Topic Segmentation

1. [2008 - Bayesian Unsupervised Topic Segmentation](./related_works/topic_segmentation/2008_Bayesian_Unsupervised_Topic_Segmentation.pdf)
   - ToDo
1. [2013 - Topic Segmentation with a Structured Topic Model](./related_works/topic_segmentation/2013_Topic_Segmentation_with_a_Structured_Topic_Model.pdf) 
   - ToDo
1. [2021 - Improving Unsupervised Dialogue Topic Segmentation with Utterance-pair Coherence scoring](./related_works/topic_segmentation/2021_Improving_Unsupervised_Dialogue_Topic_segmentation_with_Utterance-pair_Coherence_Scoring.pdf)
    - ToDo
    - Specific Loss function based on coherence computations (c minus, c plus, greater, less, ..)
    - computing means of peaks, generating graphs
    - BERT with Depth Score Computing (the meaning part)
    - [YouTube Summarization](https://www.youtube.com/watch?v=i8WUJZk5_I8&ab_channel=TechViz-TheDataScienceGuy)
1. [2021 - Unsupervised Topic Segmentation of Meetings with BERT Embeddings](./related_works/topic_segmentation/2021_Unsupervised_Topic_Segmentation_of_Meetings_with_BERT_Embeddings.pdf)
    - ToDo
    - BERT + RoBERTa + Sentence-BERT
    - outpus of 0001000000100 sequence (1 means topic change)
    - 4 stage embedding computation, sin and cosine similarity
    - softmax layer
    - [YouTube Summarization](https://www.youtube.com/watch?v=uIdqcGNoI_o&ab_channel=TechViz-TheDataScienceGuy)

### Word segmentation
1. [2006 - Contxtual Dependencies in Unsupervised Word Segmentation](./related_works/word_segmentation/2006_Contextual_Dependencies_in_Unsupervised_Word_Segmentation.pdf)
    - ToDo
1. [2009 - Bayesian Unsupervised Word Segmentation with Nested Pitman-Yor Language Modeling](./related_works/word_segmentation/2009_Bayesian_Unsupervised_Word_Segmentation_with_Nested_Pitman-Yor_Language_Modeling.pdf)
    - ToDo
1. [2009 - A Bayesian framework for word segmentation: Exploring the effects of context](./related_works/word_segmentation/2009_A_Bayesian_framework_for_word_segmentation_Exploring_the_effects_of_context.pdf)
    - Pitman-Yor, Gibs sampling, n-gram, bi-gram, chinese restaurant issue, two stage framework (generator and adapter)
    - [YouTube presentation 1:19:48](https://www.youtube.com/watch?v=3zY1xL0h16U&t=2582s&ab_channel=MicrosoftResearch)
    - ToDo 
1. [2014 - A Joint Model for Unsupervised Chinese Word Segmentation](./related_works/word_segmentation/2014_A_Joint_model_for_Unsupervised_Chinese_Word_Segmentation.pdf)
    - ToDo
1. [2022 - Unsupervised Word Segmentation with BERT Oriented Probing and Transformation](./related_works/word_segmentation/2022_Unsupervised_Word_Segmentation_with_BERT_Oriented_Probing_and_Transformation.pdf)
    - ToDo

### Melody segmentation
1. [2003 - Unsupervised Learning of Melodic Segmentation a Memory-Based Approach](./related_works/melody_segmentation/2003_Unsupervised_Learning_of_Melodic_Segmentation_A_Memory-Based_Approach.pdf)
    - ToDo
1. [2019 - Melody2Vec Distributed Representations of Melodic Phrases Based on Melody Segmentation](./related_works/melody_segmentation/2019_Melody2Vec_Distributed_Representations_of_Melodic_Phrases_based_on_Melody_Segmentation.pdf)
    - ToDo
1. [2020 - Unsupervised Melody Segmentation Based on a Nested Pitman-Yor Language Model](./related_works/melody_segmentation/2020_Unsupervised_Melody_Segmentation_Based_on_a_Nested_Pitman-Yor_Language_Model.pdf)
    - ToDo
    - from Note level (of note unigram, bigram and trigram) to Motif level (of motif unigram, bigram and trigram)
    - Combinig grouping, pitch class, duration and intervals
    - [Video BILIBILI Summarization](https://www.bilibili.com/video/BV1oD4y1R7zj/)




## Ideas
