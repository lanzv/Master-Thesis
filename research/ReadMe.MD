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
1. [Naive Bayes](./theoretical_background/2006_Naive_Bayes_classifiers.pdf)
    - Naive Baise Classifier
    - Get counts of classes from dataset ~ P(cl1)...
    - Compute probability distribution of features, in case of Gaussian NB compute Gaussian curve with its Mean and Standard Deviations
    - multiply all probabilities together (or in case of using logs sum it)
    - The class with highest probability wins
    - [YouTube NB Summarization](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
    - [YouTube Gaussian NB Summarization](https://www.youtube.com/watch?v=H3EjCKtlVog&ab_channel=StatQuestwithJoshStarmer)
1. [Latent Dirichlet Allocation](./theoretical_background/2003_Latent_Dirichlet_Allocation.pdf)
    - Task: improve tf-idf, etc.. to make data dimension lower, with less data, but with more relevant data, train the SVC classifier or whatever else
    - The task is used for topics and words
       - We get set of documents
       - Each document has its topic
       - Each document has exactly *N* words (all documents have same lenght - but there are techniques to avoid this constraints)
    - Dirichlet distribution combined with Multionimal
      - Dirichlet: triangle with 3 topics (or more and then the dimension of the triangle increases), for alpha<1, which means if the document is closer to corner, then it is more likely for the document to be writen in the specific topic/word (the one assigned to the corner)
      - Multinomial: Put probabilisitic distribution of topics/words into space, randomly choose the point, then we go with the topic/word assigned to the point
    - We try to maximize probability of generating topics of all documents correctly
      - We set parameters of dirichlets distribution (both - word and topics).
      - We get probability distribution of the document to be in specific topics (for each topic the prob based on dirichlet).
      - We get probability distribution of each topic to be mapped to specific words (for each word the prob based on dirichlet).
      - Apply multinomial distribution on both
    - Training: several options ... Laplace approximation, variational inference or Markov chain Monte Carlo (.. Gibbs Sampling)
      - trying to say how much are words mapped to specific topics and how much are topics mapped to specific documents
      - start with random distribution and iterate over all words and assign to each the most probable one (based on all other words in document and all the same words over all documents)
    - Final Step: those the most relevant for each document take and use them for training
    - [YouTube Intuition LDA](https://www.youtube.com/watch?v=T05t-SqKArY&ab_channel=Serrano.Academy)
    - [YouTube Intuition Gibbs Sampling](https://www.youtube.com/watch?v=BaM1uiCpj_E&ab_channel=Serrano.Academy)
  
### Gregorian Chants
1. [Gregorian Chants](./related_works/HILEY__David_Gregorian_Chant.pdf)


## Inspirations from other topics

### Topic Segmentation

1. [2008 - Bayesian Unsupervised Topic Segmentation](./related_works/topic_segmentation/2008_Bayesian_Unsupervised_Topic_Segmentation.pdf)
   - **BayesSeg**
     - Expected value of **posterior** distribution is multinomial distribution $\hat{\Theta}_{j}$ ... (for all models)
       - **$\hat{\Theta}_{i, j} = \frac{n_{j, i} + \Theta_{0}}{\sum^{W}_{i}n_{j, i} + W \cdot \Theta_{0}}$**, where $n_{j, i} =\sum_{t:z_{t} = j}m_{t, i}$
       - where $m_{t, i}$ is count of words *i* in sentence *t*, and the *W* is set of vocabulary words indices
     - Therefore for the language model
       - $\hat{\Theta^{j}}$ we have $p(\{x_{t}:z_{t} = j\} | \hat{\Theta}_{j}) = \prod_{t:z_{t} = j} \prod_{i \in x_{t}} \hat{\Theta}_{j, i} = \prod_{t:z_{t} = j} \prod^{W}_{i} \hat{\Theta}^{m_{t, i}}_{j, i} = \prod^{W}_{i} \hat{\Theta}^{n_{j, i}}_{j, i}$
       - the $x_{t}$ stands for sentence $t$
     - We would like to consider all models
       - $\rightarrow p(X|z, \Theta_{0}) = \prod_{j}^{k} \prod_{t:z_{t} = j} p(x_{t} | \Theta_{0}) = \prod^{K}_{j} \int d \Theta_{j} \prod_{t:z_{t} = j} p(x_{t}|\Theta_{j}) p(\Theta_{j} | \Theta_{0}) = \prod^{K}_{j} p_{dcm}(\{x_{t}:z_{t} = j\} | \Theta_{0})$
       - .. we call it marginalizing the language model
       - $p_{dcm}(\{x_{t}:z_{t} = j\} | \Theta_{0}) = \frac{\Gamma(W\cdot \Theta_{0})}{\Gamma(N_{j} + W \Theta_{0})} \cdot \prod^{K}_{i} \frac{\Gamma(n_{i, j} + W \Theta_{0})}{\Gamma(\Theta_{0})}$
     - HOW TO GET $\Theta_{0}$
       -  prior $\Theta_{0}$ is computed via gradient-based search in Viterbi EM framework
          -  E-Step ... estimating $z$ .. $p(x, z|\Theta_{0}) = p(X|z, \Theta_{0}) \cdot p(z)$
          -  M-step ... maximize $p(\Theta_{0}, X, \hat{z})$ .. $p(X|\Theta_{0}, \hat{z})\cdot p(\Theta_{0})$
     - Finally, we optimize objective function
       - $p(X, z |\Theta_{0}) = p(X|z, \Theta_{0}) \cdot p(z)$
       - $p(z)$... uniform distribution over **VALID** segmentations
       - $p(X|z, \Theta_{0})$ ... marginilized models ...
       - Lets omptimize it! **DYNAMIC PROGRAMMING**
         - $\Beta(t)$... value of the objective function for the optimal segmentation up to sentence $t$
         - $b(t', t)=p(\{x_{t'} ... x_{t} \} | z_{t'...t} = j\})$ ... contribution to the objective function from single segment between sentences $t'$ and $t$
         - Maximum value given by recurrence $\Beta(t) = max_{t'<t} \Beta(t')b(t'+1, t)$, $\Beta(0) = 1$
         - Table of size $T$, finding optimal solution dynamically
   - **BayesSeg - Cue**
     - extending *K* models by one new $\phi$
     - cannot use here the marginization
     - Using **Metropolis-Hastings** algorithm (kind of sampling, similar to Gibbs Sampling - Markov Chain), optimize
       - $p(X|z, \Theta, \phi) = \prod_{t:z_{t} \neq z_{t-1}} \prod_{i\in x^{(l)}_{t}} \phi_{i} \prod_{i\in \tilde{x}} \Theta_{z_{t, i}} \times \prod_{t:z_{t} = z_{t-1}} \prod_{i\in x_{t}} \Theta_{z_{t}, i}$
     - Proposal distribution to speed up convergence .. Basic and Linguistic one
     - Priors get from MOnte Carlo Expectation-Maximization
     - Good approach only in case that there are clear boundary words
   -  Still State of the art on the ICSI dataset, better than Facebook's 2022 BERT approach
      -  Pk score is 0.258
      -  WindowDiff score is 0.312
      -  Both scores measured comparing of two sentence topics at bounderies of sliding window through all sentences
1. [2013 - Topic Segmentation with a Structured Topic Model](./related_works/topic_segmentation/2013_Topic_Segmentation_with_a_Structured_Topic_Model.pdf) 
   - ToDo
1. [2021 - Improving Unsupervised Dialogue Topic Segmentation with Utterance-pair Coherence scoring](./related_works/topic_segmentation/2021_Improving_Unsupervised_Dialogue_Topic_segmentation_with_Utterance-pair_Coherence_Scoring.pdf)
    - ToDo
    - Specific Loss function based on coherence computations (c minus, c plus, greater, less, ..)
    - computing means of peaks, generating graphs
    - BERT with Depth Score Computing (the meaning part)
    - [YouTube Summarization](https://www.youtube.com/watch?v=i8WUJZk5_I8&ab_channel=TechViz-TheDataScienceGuy)
1. [2021 - Unsupervised Topic Segmentation of Meetings with BERT Embeddings](./related_works/topic_segmentation/2021_Unsupervised_Topic_Segmentation_of_Meetings_with_BERT_Embeddings.pdf)
    - The input to the model *M* utterances where each utterance is mapped to some of *N* topics
    - The output from the model is sequence of 0s and 1s to each utterance, where 1 means segment boundary
    - Two Approaches of sentence representation were experimented with
      - **BERT**
        - More specifically RoBERTa to pretraining
        - no fine-tuning, final vectors exrtacted via **Max Pooling**
          - robust to noisy speech data (miss-transcriptions, disfluencies, turn-taking)
          - extracting words with high semantic value from a given utterance
      - **Sentence-BERT**
        - state-of-the-art in sentence representation
        - fixed size sentence embeddings extracted using a mean over all the output vectors
    - Segmentation via four steps finding similarities
      1. BERT embeddings for every utterance
      2. Divide meetings into blocks of utterances and perform block-wise max pooling operation to extract the embedding for each block
      3. Compute cosine similarity between adjacent blocks
      4. Get topic boundaries as pairs of blocks with semantic similarity lower than a certain treshold
    - **NOT STATE OF THE ART**
      - ICSI Pk score ... 0.337 and 0.336 generated by BERT and Sentence-BERT respectivaly
      - ICSI WindowDiff ... 0.345 and 0.349
    - [YouTube Summarization](https://www.youtube.com/watch?v=uIdqcGNoI_o&ab_channel=TechViz-TheDataScienceGuy)

### Word segmentation
1. [2006 - Contxtual Dependencies in Unsupervised Word Segmentation](./related_works/word_segmentation/2006_Contextual_Dependencies_in_Unsupervised_Word_Segmentation.pdf)
    - ToDo
1. [2009 - Bayesian Unsupervised Word Segmentation with Nested Pitman-Yor Language Modeling](./related_works/word_segmentation/2009_Bayesian_Unsupervised_Word_Segmentation_with_Nested_Pitman-Yor_Language_Modeling.pdf)
    - Using here Bayesian language model based on the Pitman-Yor process (Generalization of the Dirichlet process)
    - Pitman-Yor is stochastic process generating distribution G similar to some previous distribution $G_{0}$
      - $G_{0}$ is called *base measure*
      - we can imagine base measure $G_{1}$ as a unigram distribution
      - than we can imagine $G_{2}$ as a bigram distribution given the first word *v*.
      - Than we can generate $G_{2}$ from $G_{1}$ ... $G_{2} \sim PY (G_{1}, d, \theta)$
      - We can do the same with trigram
      - We will get tree representations of n-grams
      - This process is extended Chinese Restaurant process to compute p(w|h) probabilities
    - Nested Pitman-Yor Language model
      - Specifing priors $G_{0}$
      - It is character based tree strucutre, based on the Chinese Restaurant process
      - Considering probabilities of prefixes and sufixes of words
    - Inference
      - apply ***Blocked Gibbs sampling*** to get some init tables/probs
      - apply ***Forward Filtering*** to precompute probabilities $\alpha[t][k]$ (in bigram case) of a string $c_{1}, ..., c_{t}$ with the final $k$ characters being a word (segmentations before the final $k$ characters are marginalized  using some recursive calculation)
      - apply ***Backward Sampling*** .. start at the end of the sentence and dynamically find the best boundary of the last word, after that the prelast word, after that the word before that previous word, etc...
      - apply ***Poisson correction*** to work better with long words
    - Finally two models are used, bigram and trigram based
      - MSR dataset accuracy score NPY(2) 80.2
      - MSR dataset accuracy score NPY(3) 80.7
2. [2009 - A Bayesian framework for word segmentation: Exploring the effects of context](./related_works/word_segmentation/2009_A_Bayesian_framework_for_word_segmentation_Exploring_the_effects_of_context.pdf)
    - Pitman-Yor, Gibs sampling, n-gram, bi-gram, chinese restaurant issue, two stage framework (generator and adapter)
    - [YouTube presentation 1:19:48](https://www.youtube.com/watch?v=3zY1xL0h16U&t=2582s&ab_channel=MicrosoftResearch)
    - ToDo 
3. [2014 - A Joint Model for Unsupervised Chinese Word Segmentation](./related_works/word_segmentation/2014_A_Joint_model_for_Unsupervised_Chinese_Word_Segmentation.pdf)
    - ToDo
4. [2022 - Unsupervised Word Segmentation with BERT Oriented Probing and Transformation](./related_works/word_segmentation/2022_Unsupervised_Word_Segmentation_with_BERT_Oriented_Probing_and_Transformation.pdf)
    - The final model consists of three parts
      - **Generative module**
        - suggests the plausible word boundaries by **probing** BERT
        - lets say we have the input sequence, sentence for instance
        - lets mask one by one of tokens, so when we mask ith token, we will get at that ith postion by BERT its representation $H_{i}$
        - We will get all these representations for all tokens
        - After that we will do the same, but now we mask each successive pair of tokens i, j.
        - The representation given by BERT at ith position is $H_{i,j}$, the representation at jth position is $H_{j, i}$
        - If twho token $x_{i}$ and $x_{j}$ are close enough, they should be part of one word
        - lets define distance $d = \frac{(|H_{i,j} - H_{i}| + |H_{j, i} - H_{j}|)}{2}$, and if the distnace $d$ is lower then some trehshold, than it is one word, otherwise we segment them
      - **Discriminative module**
        - **transforms** the implicit boundary information into explicit sequence labels
        - starts with segmentation by Generative module, which is not good for large words
        - fine-tuning BERT to predict *Beginning* and *Inside* lables
        - additional output layer peojcting the previous representation into **BI** labels
        - only considering words with really high confidence
      - **Evaluation module**
        - estimates the performance of the model in an unsupervise manner
        - training Generative module from the Discriminative module based on loss $||d-treshold||^{2}$
        - training Discriminative module using MLM (Masked-Language Modeling)
    - **State-of-the-art** at MSR dataset
      - MSR dataset accuracy score 83.0

### Melody segmentation
1. [2003 - Unsupervised Learning of Melodic Segmentation a Memory-Based Approach](./related_works/melody_segmentation/2003_Unsupervised_Learning_of_Melodic_Segmentation_A_Memory-Based_Approach.pdf)
    - ToDo
1. [2019 - Melody2Vec Distributed Representations of Melodic Phrases Based on Melody Segmentation](./related_works/melody_segmentation/2019_Melody2Vec_Distributed_Representations_of_Melodic_Phrases_based_on_Melody_Segmentation.pdf)
    - ToDo
    - There is a **Grouping Preference Rules** in GTTM (1983 - Lerdahl, Jackendoff - A Generative Theory of Tonal Music) that contains some rules for melody segmentation
    - inspiration of Mikolov's Word2Vec, skip-gram was used, the idea is that melodies encoded in vecotrs should help in another shared-tasks
1. [2020 - Unsupervised Melody Segmentation Based on a Nested Pitman-Yor Language Model](./related_works/melody_segmentation/2020_Unsupervised_Melody_Segmentation_Based_on_a_Nested_Pitman-Yor_Language_Model.pdf)
    - ToDo
    - Inspired by the word segmentation 2009 article - Nested Pitman-Yor language model
    - from Note level (of note unigram, bigram and trigram) to Motif level (of motif unigram, bigram and trigram)
    - Combinig grouping, pitch class, duration and intervals (durations gives the best score - could be problem with gregorian chants)
    - **EVALUATED** on Grouping Structure of GTTM
      - 1983 - Lerdahl, Jackendoff - A Generative Theory of Tonal Music
      - there are several GPRs (Grouping Preference Rules)
      - the NPYLM model is evaluated on gold data generated by these rules.
      - probabily not the best method to evaluate, but good enough
    - [Video BILIBILI Summarization](https://www.bilibili.com/video/BV1oD4y1R7zj/)



## Example Source Codes

### Topic Model based project
 - [Pitman-Yor Process](https://github.com/vchahun/vpyp)
   - Implementation of a topic model (LDA with a Pitman-Yor prior instead of a Dirichlet prior)
   - python 2.7
 - [Topic Segmentation](https://github.com/ldulcic/text-segmentation)
   - based on the LDA topic model
   - implementation of TextTiling Algorithm described in [this paper](https://aclanthology.org/W12-3307.pdf) (2012)

### Bayesian
 - [Bayesian Unsupervised Topic Segmentation](https://github.com/jacobeisenstein/bayes-seg)
   - original source code of its paper
   - Java
   - There is a [python code](https://github.com/rohitmanral/Bayesian-Unsupervised-Topic-Segmentation_Replica) of the paper, not original, hard to say if its complete or not

### BERT
 - [Unsupervised Topic Segmentation of Meetings with BERT Embeddings](https://github.com/gdamaskinos/unsupervised_topic_segmentation)
   - original implementation of the original paper
   - python (pytorch)
   - based on huggingface pretrained roberta model
   - [Reimplementation](https://github.com/Feodoros/Unsupervised-Topic-Segmentation-of-Meetings) and comparision with other approaches
 - [Unsupervised Chinese Word Segmentation with BERT Oriented Probing and Transformation](https://github.com/QbethQ/Unsupervised_CWS_BOPT)
   - some kind of [repository copy](https://github.com/liweitj47/BERT_unsupervised_word_segmentation) or the not final version.
   - original implementation of the original paper
   - python (pytorch)
   - based on huggingface pretrained roberta model

## Ideas
