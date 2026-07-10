# QTA Lab 05 Answers: Supervised classification with House of Commons speeches


## Learning goals

In this lab, you will learn how to:

- define a supervised classification task;
- create training and test sets;
- train a Naive Bayes classifier using a DFM;
- train a regularised logistic regression classifier;
- evaluate classifiers with confusion matrices, precision, recall, and
  F1;
- inspect predictive features;
- think about how human and LLM-assisted labels fit into supervised text
  analysis.

The running example is a House of Commons dataset containing speeches by
Jeremy Corbyn and Theresa May during their time as party leaders. We
will train classifiers to predict whether a speech was delivered by
Corbyn or May. This is a useful teaching task because the labels already
exist as metadata, but it is not a perfect research design. The model
may learn speaker style, government/opposition roles, parliamentary
offices, time period, or policy topics. That is part of the lesson:
supervised classification can work well while still requiring careful
interpretation.

## Load packages

``` r
library(dplyr)
library(stringr)
library(ggplot2)
library(quanteda)
library(quanteda.textmodels)
library(quanteda.textstats)
library(quanteda.textplots)
library(caret)
```

## Read and prepare the data

``` r
data_path <- "Data/hc_corbyn_may_party_leaders_2015_2020.rds"

leader_speeches <- readRDS(data_path)

leader_speeches <- leader_speeches |>
  mutate(
    year = as.integer(format(date, "%Y")),
    speaker_label = if_else(speaker == "Jeremy Corbyn", "Corbyn", "May")
  )
```

The full leader-period dataset contains many more Theresa May speeches
than Jeremy Corbyn speeches. If we train on the unbalanced dataset, a
classifier can look good simply by predicting the majority class. For
the main example, we therefore create a balanced dataset with the same
number of speeches for each speaker.

``` r
leader_speeches |>
  count(speaker, party)
```

    # A tibble: 2 × 3
      speaker       party            n
      <chr>         <chr>        <int>
    1 Jeremy Corbyn Labour        1454
    2 Theresa May   Conservative  6997

``` r
min_speaker_n <- leader_speeches |>
  count(speaker_label) |>
  summarise(min_n = min(n)) |>
  pull(min_n)

set.seed(20260710)

leader_balanced <- leader_speeches |>
  group_by(speaker_label) |>
  slice_sample(n = min_speaker_n) |>
  ungroup()

leader_balanced |>
  count(speaker, party)
```

    # A tibble: 2 × 3
      speaker       party            n
      <chr>         <chr>        <int>
    1 Jeremy Corbyn Labour        1454
    2 Theresa May   Conservative  1454

## Create a corpus and DFM

``` r
leader_corpus <- corpus(
  leader_balanced,
  text_field = "text"
)

ndoc(leader_corpus)
```

    [1] 2908

We tokenize the corpus, remove punctuation, numbers, URLs, and English
stopwords, and lower-case all tokens.

``` r
leader_tokens <- tokens(
  leader_corpus,
  what = "word",
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE,
  remove_url = TRUE,
  remove_separators = TRUE,
  split_hyphens = FALSE
) |>
  tokens_tolower() |>
  tokens_remove(stopwords("en"))
```

We then create a DFM and trim it so that we only keep features that
appear in at least 1 percent of documents.

``` r
leader_dfm <- dfm(leader_tokens) |>
  dfm_trim(
    min_docfreq = 0.01,
    docfreq_type = "prop"
  )

dim(leader_dfm)
```

    [1] 2908 1066

``` r
topfeatures(leader_dfm, 30)
```

         prime   minister government     people        hon       deal      house 
          2712       2502       2079       1958       1847       1530       1498 
             â      right    country        can   european      union     friend 
          1325       1300       1180       1067        948        914        897 
          said         us        now       want       many       work       last 
           892        869        683        651        641        636        628 
          also     ensure       need        one       time     future      clear 
           620        609        591        589        574        569        559 
          just    support 
           554        553 

## Create training and test sets

Supervised learning requires labelled examples. We split the data into:

- a training set used to estimate the model;
- a test set held back for evaluation.

The test set should not be used to make modelling decisions. It
simulates new unseen data.

The code below first fixes the random seed so that everyone gets the
same split. It then gives each speech a simple numeric ID, randomly
selects 70% of those IDs for the training set, and keeps the remaining
30% for the test set. The two `dfm_subset()` calls apply that split to
the document-feature matrix, while the two `table()` calls check how
many Corbyn and May speeches ended up in each set. This matters because
a very uneven split can make model evaluation misleading.

``` r
set.seed(20260710)

docvars(leader_dfm, "id_numeric") <- seq_len(ndoc(leader_dfm))

train_ids <- sample(
  seq_len(ndoc(leader_dfm)),
  size = floor(0.70 * ndoc(leader_dfm)),
  replace = FALSE
)

train_dfm <- dfm_subset(
  leader_dfm,
  id_numeric %in% train_ids
)

test_dfm <- dfm_subset(
  leader_dfm,
  !id_numeric %in% train_ids
)

table(docvars(train_dfm, "speaker_label"))
```


    Corbyn    May 
      1005   1030 

``` r
table(docvars(test_dfm, "speaker_label"))
```


    Corbyn    May 
       449    424 

Check that there is no overlap between the training and test sets.

``` r
intersect(
  docvars(train_dfm, "id_numeric"),
  docvars(test_dfm, "id_numeric")
)
```

    integer(0)

## Naive Bayes classifier

Naive Bayes is a simple and often surprisingly effective text
classifier. It estimates the probability of features given each class.

In the code below, `train_dfm` supplies the word features and `y`
supplies the known speaker labels for the training speeches. We set
`smooth = 1` so that words not seen with one speaker in the training set
do not receive a probability of zero. The `prior = "docfreq"` option
estimates the prior probability of each speaker from the training
documents. Finally, `distribution = "multinomial"` tells the model to
treat the DFM as word-count data, which is the standard Naive Bayes
setup for text classification.

``` r
speaker_classifier_nb <- textmodel_nb(
  train_dfm,
  y = docvars(train_dfm, "speaker_label"),
  smooth = 1,
  prior = "docfreq",
  distribution = "multinomial"
)

summary(speaker_classifier_nb)
```


    Call:
    textmodel_nb.dfm(x = train_dfm, y = docvars(train_dfm, "speaker_label"), 
        smooth = 1, prior = "docfreq", distribution = "multinomial")

    Class Priors:
    (showing first 2 elements)
    Corbyn    May 
    0.4939 0.5061 

    Estimated Feature Scores:
            evidence      tory    party government    really    think   workers
    Corbyn 0.0007143 9.783e-04 0.002655   0.016367 0.0010249 0.002391 0.0032611
    May    0.0004670 2.594e-05 0.002179   0.009339 0.0002854 0.003035 0.0007005
             rights      see yesterday   private      bill      rule     parts
    Corbyn 0.003261 0.001584  0.001522 0.0009162 0.0014131 0.0007299 0.0003727
    May    0.001375 0.003943  0.000441 0.0003373 0.0006745 0.0002594 0.0006486
           international   labour organisation convention   talking      hon
    Corbyn      0.001615 0.003401    0.0002950  0.0004503 0.0005590 0.004581
    May         0.001479 0.002983    0.0002075  0.0002075 0.0006226 0.025217
             friend   member     great   protect european  country     real
    Corbyn 0.002671 0.003354 0.0011957 0.0012113 0.003634 0.009069 0.001258
    May    0.012349 0.002179 0.0007783 0.0003891 0.011545 0.005604 0.000467
               prime minister    threat
    Corbyn 0.0286043 0.026042 0.0009939
    May    0.0008561 0.001427 0.0002075

Before prediction, we match the test DFM to the training DFM so that
both have the same feature columns in the same order.

``` r
matched_test_dfm <- dfm_match(
  test_dfm,
  features = featnames(train_dfm)
)

pred_speaker_nb <- predict(
  speaker_classifier_nb,
  newdata = matched_test_dfm,
  type = "class"
)

head(pred_speaker_nb)
```

     text2  text3  text7  text8  text9 text11 
    Corbyn Corbyn Corbyn Corbyn Corbyn Corbyn 
    Levels: Corbyn May

We can also inspect predicted class probabilities.

``` r
pred_prob_nb <- predict(
  speaker_classifier_nb,
  newdata = matched_test_dfm,
  type = "probability"
)

head(pred_prob_nb)
```

            
    docs        Corbyn           May
      text2  1.0000000  1.704344e-15
      text3  1.0000000  1.614109e-17
      text7  1.0000000  1.253606e-22
      text8  1.0000000  5.498114e-10
      text9  0.9999992  7.507746e-07
      text11 1.0000000 2.927665e-102

## Evaluate the classifier

A confusion matrix compares predicted labels to true labels.

``` r
tab_class_nb <- table(
  predicted_speaker = pred_speaker_nb,
  actual_speaker = docvars(test_dfm, "speaker_label")
)

tab_class_nb
```

                     actual_speaker
    predicted_speaker Corbyn May
               Corbyn    411  25
               May        38 399

The `caret::confusionMatrix()` function reports accuracy, precision,
recall, and F1.

``` r
confusionMatrix(
  tab_class_nb,
  mode = "prec_recall"
)
```

    Confusion Matrix and Statistics

                     actual_speaker
    predicted_speaker Corbyn May
               Corbyn    411  25
               May        38 399
                                              
                   Accuracy : 0.9278          
                     95% CI : (0.9086, 0.9441)
        No Information Rate : 0.5143          
        P-Value [Acc > NIR] : <2e-16          
                                              
                      Kappa : 0.8557          
                                              
     Mcnemar's Test P-Value : 0.1306          
                                              
                  Precision : 0.9427          
                     Recall : 0.9154          
                         F1 : 0.9288          
                 Prevalence : 0.5143          
             Detection Rate : 0.4708          
       Detection Prevalence : 0.4994          
          Balanced Accuracy : 0.9282          
                                              
           'Positive' Class : Corbyn          
                                              

High performance is plausible here because Corbyn and May often spoke
from different institutional positions, in different policy contexts,
and with different rhetorical styles. But that also means we should be
careful: the classifier may be learning more than “speaker style”. It
may also learn government versus opposition language, Prime Minister’s
Questions, Brexit-era topics, or procedural patterns.

## Inspect predictive features

For Naive Bayes, we can inspect features that are more strongly
associated with one class than another.

``` r
feature_probs <- as.data.frame(t(speaker_classifier_nb$param))
feature_probs$feature <- rownames(feature_probs)

epsilon <- 1e-10

feature_probs <- feature_probs |>
  mutate(
    log_odds_corbyn = log((Corbyn + epsilon) / (May + epsilon))
  )

top_corbyn <- feature_probs |>
  arrange(desc(log_odds_corbyn)) |>
  select(feature, Corbyn, May, log_odds_corbyn) |>
  head(20)

top_may <- feature_probs |>
  arrange(log_odds_corbyn) |>
  select(feature, Corbyn, May, log_odds_corbyn) |>
  head(20)

top_corbyn
```

                  feature       Corbyn          May log_odds_corbyn
    crisis         crisis 0.0031678986 7.782909e-05        3.706307
    tory             tory 0.0009783216 2.594303e-05        3.629932
    prime           prime 0.0286042611 8.561200e-04        3.508900
    wrote           wrote 0.0006522144 2.594303e-05        3.224467
    failed         failed 0.0018324120 7.782909e-05        3.158872
    copy             copy 0.0005900988 2.594303e-05        3.124383
    promised     promised 0.0011180819 5.188606e-05        3.070318
    urgent         urgent 0.0005590409 2.594303e-05        3.070316
    arms             arms 0.0005124542 2.594303e-05        2.983305
    decent         decent 0.0004813964 2.594303e-05        2.920784
    minister     minister 0.0260419902 1.426867e-03        2.904229
    alone           alone 0.0004658674 2.594303e-05        2.887995
    œa                 œa 0.0004503385 2.594303e-05        2.854093
    stress         stress 0.0004503385 2.594303e-05        2.854093
    apparently apparently 0.0004348096 2.594303e-05        2.819002
    red               red 0.0008075036 5.188606e-05        2.744896
    rising         rising 0.0007764457 5.188606e-05        2.705675
    wonder         wonder 0.0003882229 2.594303e-05        2.705673
    defeat         defeat 0.0003882229 2.594303e-05        2.705673
    unable         unable 0.0003726940 2.594303e-05        2.664851

``` r
top_may
```

                          feature       Corbyn          May log_odds_corbyn
    raises                 raises 4.658674e-05 0.0014009236       -3.403569
    kingdom               kingdom 2.329337e-04 0.0045919162       -2.981298
    relation             relation 1.863470e-04 0.0030872205       -2.807416
    gentleman           gentleman 5.124542e-04 0.0079126239       -2.737003
    enable                 enable 4.658674e-05 0.0005707466       -2.505627
    various               various 6.211566e-05 0.0006745188       -2.385000
    certain               certain 4.658674e-05 0.0004929176       -2.359024
    ensuring             ensuring 2.174048e-04 0.0021013854       -2.268591
    implementation implementation 1.242313e-04 0.0011933793       -2.262398
    particular         particular 1.708181e-04 0.0015306387       -2.192841
    scottish             scottish 1.552891e-04 0.0013490375       -2.161857
    lady                     lady 3.105783e-04 0.0025424169       -2.102434
    looked                 looked 7.764457e-05 0.0005966897       -2.039254
    independent       independent 1.708181e-04 0.0012712084       -2.007124
    encourage           encourage 7.764457e-05 0.0005707466       -1.994803
    recognise           recognise 4.969253e-04 0.0035541950       -1.967444
    discussions       discussions 2.174048e-04 0.0015306387       -1.951679
    friend's             friend's 9.317349e-05 0.0006485757       -1.940315
    important           important 1.087024e-03 0.0073937633       -1.917193
    gentleman's       gentleman's 7.764457e-05 0.0005188606       -1.899492

This is a useful diagnostic step. If predictive features are mostly
artefacts, names, or procedural terms, that tells us something about the
classifier.

## Logistic regression classifier

Let us try a regularised logistic regression classifier.

Logistic regression takes a different route from Naive Bayes. Instead of
modelling how likely words are within each class, it learns weights for
features that push a speech toward one label or the other. Words with
positive weights provide evidence for one speaker; words with negative
weights provide evidence for the other. The regularisation part keeps
the model from relying too heavily on too many small, noisy word
patterns, which is especially useful when the DFM has thousands of
features.

``` r
speaker_classifier_lr <- textmodel_lr(
  train_dfm,
  y = docvars(train_dfm, "speaker_label")
)

pred_speaker_lr <- predict(
  speaker_classifier_lr,
  newdata = matched_test_dfm,
  type = "class"
)

tab_class_lr <- table(
  predicted_speaker = pred_speaker_lr,
  actual_speaker = docvars(test_dfm, "speaker_label")
)

tab_class_lr
```

                     actual_speaker
    predicted_speaker Corbyn May
               Corbyn    402  13
               May        47 411

``` r
confusionMatrix(
  tab_class_lr,
  mode = "prec_recall"
)
```

    Confusion Matrix and Statistics

                     actual_speaker
    predicted_speaker Corbyn May
               Corbyn    402  13
               May        47 411
                                              
                   Accuracy : 0.9313          
                     95% CI : (0.9124, 0.9471)
        No Information Rate : 0.5143          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.8627          
                                              
     Mcnemar's Test P-Value : 2.042e-05       
                                              
                  Precision : 0.9687          
                     Recall : 0.8953          
                         F1 : 0.9306          
                 Prevalence : 0.5143          
             Detection Rate : 0.4605          
       Detection Prevalence : 0.4754          
          Balanced Accuracy : 0.9323          
                                              
           'Positive' Class : Corbyn          
                                              

We can inspect the logistic regression coefficients. In this binary
model, positive coefficients point toward the class shown in the
coefficient column. Negative coefficients point toward the other class.

``` r
lr_coefficients <- coef(speaker_classifier_lr)
lr_coefficients <- as.matrix(lr_coefficients)

head(lr_coefficients)
```

                        May
    (Intercept)  0.22996947
    evidence     0.00000000
    tory        -0.53144464
    party        0.00000000
    government  -0.08710753
    really       0.00000000

``` r
positive_speaker <- colnames(lr_coefficients)[1]
negative_speaker <- setdiff(
  levels(docvars(train_dfm, "speaker_label")),
  positive_speaker
)

lr_features <- data.frame(
  feature = rownames(lr_coefficients),
  coefficient = as.numeric(lr_coefficients[, 1])
) |>
  filter(feature != "(Intercept)")

cat("Features with positive coefficients point toward:", positive_speaker, "\n")
```

    Features with positive coefficients point toward: May 

``` r
lr_features |>
  arrange(desc(coefficient)) |>
  head(20)
```

             feature coefficient
    1      providing   1.4380145
    2       ensuring   1.3043497
    3       relation   1.2989987
    4        kingdom   1.2970970
    5       meetings   1.2516124
    6       interest   1.2063857
    7     delivering   1.1977684
    8            hon   1.1692476
    9        deliver   0.9907268
    10      movement   0.9645512
    11     important   0.9515092
    12       earlier   0.9050535
    13   discussions   0.8731203
    14      position   0.8372404
    15     recognise   0.7764154
    16 conservatives   0.7676440
    17     questions   0.7619971
    18        summer   0.7370714
    19          able   0.7302320
    20     encourage   0.7270833

``` r
cat("Features with negative coefficients point toward:", negative_speaker, "\n")
```

    Features with negative coefficients point toward: 

``` r
lr_features |>
  arrange(coefficient) |>
  head(20)
```

            feature coefficient
    1         prime  -2.2375615
    2  intervention  -1.7864570
    3       serious  -1.4936495
    4          away  -1.4619108
    5        decent  -1.3070092
    6          œthe  -1.1598867
    7       speaker  -1.1559899
    8        surely  -1.1306093
    9          wait  -1.1304298
    10       fiscal  -1.0339306
    11         lack  -0.9967643
    12       either  -0.9772247
    13          six  -0.9291708
    14         gets  -0.8771924
    15     priority  -0.8318555
    16   industries  -0.8186744
    17      opposed  -0.7781578
    18       member  -0.7414464
    19        thank  -0.7403421
    20     promised  -0.7152356

## TF-IDF weighting

So far, the DFM contains raw counts. We can also weight features using
term frequency-inverse document frequency, or TF-IDF.

``` r
train_dfm_tfidf <- dfm_tfidf(train_dfm)
matched_test_dfm_tfidf <- dfm_tfidf(matched_test_dfm)

speaker_classifier_lr_tfidf <- textmodel_lr(
  train_dfm_tfidf,
  y = docvars(train_dfm_tfidf, "speaker_label")
)

pred_speaker_lr_tfidf <- predict(
  speaker_classifier_lr_tfidf,
  newdata = matched_test_dfm_tfidf,
  type = "class"
)

tab_class_lr_tfidf <- table(
  predicted_speaker = pred_speaker_lr_tfidf,
  actual_speaker = docvars(test_dfm, "speaker_label")
)

tab_class_lr_tfidf
```

                     actual_speaker
    predicted_speaker Corbyn May
               Corbyn    398  15
               May        51 409

``` r
confusionMatrix(
  tab_class_lr_tfidf,
  mode = "prec_recall"
)
```

    Confusion Matrix and Statistics

                     actual_speaker
    predicted_speaker Corbyn May
               Corbyn    398  15
               May        51 409
                                              
                   Accuracy : 0.9244          
                     95% CI : (0.9048, 0.9411)
        No Information Rate : 0.5143          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.849           
                                              
     Mcnemar's Test P-Value : 1.646e-05       
                                              
                  Precision : 0.9637          
                     Recall : 0.8864          
                         F1 : 0.9234          
                 Prevalence : 0.5143          
             Detection Rate : 0.4559          
       Detection Prevalence : 0.4731          
          Balanced Accuracy : 0.9255          
                                              
           'Positive' Class : Corbyn          
                                              

## Human coding, LLM-assisted coding, and validation

In this lab, the label `speaker_label` was created from metadata. In
many QTA projects, the label is not available. Researchers may create
labels through:

- expert human coding;
- crowd coding;
- weak supervision;
- LLM-assisted coding;
- a combination of these.

Regardless of how labels are created, the same principles apply:

- define the coding task clearly;
- write a codebook;
- separate training and test data;
- validate against held-out labels;
- inspect errors;
- report the classifier, features, training data, and evaluation
  metrics.

LLMs can help produce candidate labels, explanations, and disagreement
checks, but those labels still need validation. In the language we use
later with `quallmer`, an LLM-assisted workflow should be
codebook-centred, produce structured coding outputs, and preserve enough
provenance to inspect what happened.

## Practice exercises

1.  Create an unbalanced DFM using all Corbyn and May leader-period
    speeches. Call it `leader_unbalanced_dfm`. Compare the number of
    speeches by speaker.

``` r
leader_unbalanced_corpus <- corpus(
  leader_speeches,
  text_field = "text"
)

leader_unbalanced_tokens <- tokens(
  leader_unbalanced_corpus,
  what = "word",
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE,
  remove_url = TRUE,
  remove_separators = TRUE,
  split_hyphens = FALSE
) |>
  tokens_tolower() |>
  tokens_remove(stopwords("en"))

leader_unbalanced_dfm <- dfm(leader_unbalanced_tokens) |>
  dfm_trim(
    min_docfreq = 0.01,
    docfreq_type = "prop"
  )

dim(leader_unbalanced_dfm)
```

    [1] 8451  868

``` r
table(docvars(leader_unbalanced_dfm, "speaker_label"))
```


    Corbyn    May 
      1454   6997 

2.  Create a 70/30 train-test split for `leader_unbalanced_dfm`.

``` r
set.seed(20260710)

docvars(leader_unbalanced_dfm, "id_numeric") <-
  seq_len(ndoc(leader_unbalanced_dfm))

unbalanced_train_ids <- sample(
  seq_len(ndoc(leader_unbalanced_dfm)),
  size = floor(0.70 * ndoc(leader_unbalanced_dfm)),
  replace = FALSE
)

unbalanced_train_dfm <- dfm_subset(
  leader_unbalanced_dfm,
  id_numeric %in% unbalanced_train_ids
)

unbalanced_test_dfm <- dfm_subset(
  leader_unbalanced_dfm,
  !id_numeric %in% unbalanced_train_ids
)

table(docvars(unbalanced_train_dfm, "speaker_label"))
```


    Corbyn    May 
      1031   4884 

``` r
table(docvars(unbalanced_test_dfm, "speaker_label"))
```


    Corbyn    May 
       423   2113 

``` r
intersect(
  docvars(unbalanced_train_dfm, "id_numeric"),
  docvars(unbalanced_test_dfm, "id_numeric")
)
```

    integer(0)

3.  Train a Naive Bayes classifier on the unbalanced training set and
    predict the speaker labels in the unbalanced test set.

``` r
unbalanced_classifier_nb <- textmodel_nb(
  unbalanced_train_dfm,
  y = docvars(unbalanced_train_dfm, "speaker_label"),
  smooth = 1,
  prior = "docfreq",
  distribution = "multinomial"
)

unbalanced_test_matched_dfm <- dfm_match(
  unbalanced_test_dfm,
  features = featnames(unbalanced_train_dfm)
)

unbalanced_pred_speaker_nb <- predict(
  unbalanced_classifier_nb,
  newdata = unbalanced_test_matched_dfm,
  type = "class"
)

head(unbalanced_pred_speaker_nb)
```

     text2  text7 text12 text21 text23 text27 
    Corbyn Corbyn Corbyn Corbyn Corbyn Corbyn 
    Levels: Corbyn May

4.  Create a confusion matrix and calculate precision, recall, and F1.
    Compare the result with the balanced model above.

``` r
unbalanced_tab_class_nb <- table(
  predicted_speaker = unbalanced_pred_speaker_nb,
  actual_speaker = docvars(unbalanced_test_dfm, "speaker_label")
)

unbalanced_tab_class_nb
```

                     actual_speaker
    predicted_speaker Corbyn  May
               Corbyn    377   91
               May        46 2022

``` r
confusionMatrix(
  unbalanced_tab_class_nb,
  mode = "prec_recall"
)
```

    Confusion Matrix and Statistics

                     actual_speaker
    predicted_speaker Corbyn  May
               Corbyn    377   91
               May        46 2022
                                              
                   Accuracy : 0.946           
                     95% CI : (0.9365, 0.9545)
        No Information Rate : 0.8332          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.8136          
                                              
     Mcnemar's Test P-Value : 0.0001705       
                                              
                  Precision : 0.8056          
                     Recall : 0.8913          
                         F1 : 0.8462          
                 Prevalence : 0.1668          
             Detection Rate : 0.1487          
       Detection Prevalence : 0.1845          
          Balanced Accuracy : 0.9241          
                                              
           'Positive' Class : Corbyn          
                                              

The unbalanced model may have high overall accuracy because May has many
more speeches. Compare precision and recall by class rather than only
accuracy.

5.  Train a logistic regression classifier on the same unbalanced
    training set. Does it perform better or worse than Naive Bayes?

``` r
unbalanced_classifier_lr <- textmodel_lr(
  unbalanced_train_dfm,
  y = docvars(unbalanced_train_dfm, "speaker_label")
)

unbalanced_pred_speaker_lr <- predict(
  unbalanced_classifier_lr,
  newdata = unbalanced_test_matched_dfm,
  type = "class"
)

unbalanced_tab_class_lr <- table(
  predicted_speaker = unbalanced_pred_speaker_lr,
  actual_speaker = docvars(unbalanced_test_dfm, "speaker_label")
)

unbalanced_tab_class_lr
```

                     actual_speaker
    predicted_speaker Corbyn  May
               Corbyn    354   21
               May        69 2092

``` r
confusionMatrix(
  unbalanced_tab_class_lr,
  mode = "prec_recall"
)
```

    Confusion Matrix and Statistics

                     actual_speaker
    predicted_speaker Corbyn  May
               Corbyn    354   21
               May        69 2092
                                              
                   Accuracy : 0.9645          
                     95% CI : (0.9566, 0.9714)
        No Information Rate : 0.8332          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.8663          
                                              
     Mcnemar's Test P-Value : 7.262e-07       
                                              
                  Precision : 0.9440          
                     Recall : 0.8369          
                         F1 : 0.8872          
                 Prevalence : 0.1668          
             Detection Rate : 0.1396          
       Detection Prevalence : 0.1479          
          Balanced Accuracy : 0.9135          
                                              
           'Positive' Class : Corbyn          
                                              

6.  Inspect the most predictive Naive Bayes features for Corbyn and May
    in the unbalanced model. Which features look like speaker style?
    Which might reflect office, time period, or topic?

``` r
unbalanced_feature_probs <- as.data.frame(t(unbalanced_classifier_nb$param))
unbalanced_feature_probs$feature <- rownames(unbalanced_feature_probs)

epsilon <- 1e-10

unbalanced_feature_probs <- unbalanced_feature_probs |>
  mutate(
    log_odds_corbyn = log((Corbyn + epsilon) / (May + epsilon))
  )

unbalanced_feature_probs |>
  arrange(desc(log_odds_corbyn)) |>
  select(feature, Corbyn, May, log_odds_corbyn) |>
  head(20)
```

                  feature       Corbyn          May log_odds_corbyn
    worse           worse 0.0010431154 2.758545e-05        3.632675
    prime           prime 0.0312272336 1.031696e-03        3.410087
    crisis         crisis 0.0031293463 1.544785e-04        3.008524
    minister     minister 0.0284124776 1.566853e-03        2.897759
    cuts             cuts 0.0021193457 1.379272e-04        2.732136
    minister's minister's 0.0032286906 2.151665e-04        2.708419
    failed         failed 0.0018213127 1.655127e-04        2.398264
    war               war 0.0014901649 1.434443e-04        2.340695
    reality       reality 0.0013908206 1.379272e-04        2.310922
    poverty       poverty 0.0023677065 2.593032e-04        2.211679
    promised     promised 0.0010596728 1.213760e-04        2.166822
    seems           seems 0.0008775416 1.048247e-04        2.124834
    cut               cut 0.0025829525 3.144741e-04        2.105786
    œthe             œthe 0.0008444268 1.048247e-04        2.086368
    failure       failure 0.0011424598 1.489614e-04        2.037251
    explain       explain 0.0008444268 1.103418e-04        2.035075
    millions     millions 0.0010431154 1.379272e-04        2.023240
    waiting       waiting 0.0011093450 1.544785e-04        1.971470
    instead       instead 0.0009768859 1.599956e-04        1.809223
    human           human 0.0015895092 2.648203e-04        1.792129

``` r
unbalanced_feature_probs |>
  arrange(log_odds_corbyn) |>
  select(feature, Corbyn, May, log_odds_corbyn) |>
  head(20)
```

                          feature       Corbyn          May log_odds_corbyn
    refers                 refers 1.655739e-05 0.0004744697       -3.355359
    raises                 raises 4.967216e-05 0.0013351356       -3.291341
    kingdom               kingdom 1.986887e-04 0.0051695126       -3.258794
    duties                 duties 1.655739e-05 0.0003972304       -3.177678
    relation             relation 2.318034e-04 0.0034812833       -2.709266
    determine           determine 3.311478e-05 0.0004192988       -2.538601
    enable                 enable 4.967216e-05 0.0006123969       -2.511934
    gentleman           gentleman 7.947546e-04 0.0087445864       -2.398157
    implementation implementation 1.159017e-04 0.0012578963       -2.384453
    ensuring             ensuring 2.483608e-04 0.0021682160       -2.166777
    operate               operate 6.622955e-05 0.0005406747       -2.099690
    looked                 looked 6.622955e-05 0.0005296406       -2.079071
    activity             activity 4.967216e-05 0.0003972304       -2.079070
    various               various 8.278694e-05 0.0006510165       -2.062264
    important           important 9.934433e-04 0.0071446305       -1.972939
    importance         importance 1.821313e-04 0.0013075501       -1.971182
    arrangements     arrangements 2.814756e-04 0.0020192546       -1.970438
    delivering         delivering 1.821313e-04 0.0013020330       -1.966954
    ways                     ways 1.324591e-04 0.0009323881       -1.951475
    recognise           recognise 4.470495e-04 0.0031281896       -1.945540

7.  Write a short reflection: if you were classifying whether speeches
    are about crime and public safety, rather than classifying speaker,
    how would you obtain and validate training labels? Where might
    LLM-assisted coding help, and where would you still need human
    judgement?

``` r
# A good workflow would start with a clear codebook that defines what counts as
# a crime and public safety speech. I would hand-code a stratified sample across
# speakers, parties, years, and speech lengths, then use inter-coder checks or
# adjudication to improve the coding rules.
#
# LLM-assisted coding could help propose candidate coding rules, flag ambiguous
# cases, or pre-code a larger pool of speeches after the codebook is stable. I
# would still validate those labels against human-coded data, inspect
# disagreement cases, and report performance on a held-out validation set.
#
# Human judgement is especially important for borderline cases, such as speeches
# that mention police only in passing, procedural contributions, or speeches
# where public safety language is used metaphorically rather than as the main
# topic.
```
