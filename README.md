Materials for the Essex Summer School 2026 course *Introduction to Quantitative Text Analysis*.

Instructor: [Martijn Schoonvelde](http://mschoonvelde.com)  
Syllabus: [Syllabus_QTA.pdf](Syllabus_QTA.pdf)  
Course Slack: [essqta26.slack.com](https://essqta26.slack.com)

## Before the first lab

Please install [R](https://cran.r-project.org/), [Quarto](https://quarto.org/docs/get-started/), and [Positron](https://positron.posit.co/download.html) before the first lab.

## Slides

| Date | Slides | Date | Slides |
| --- | :---: | --- | :---: |
| July 6 | [Slides PDF](Slides/Day_1/Slides_QTA_1.pdf) | July 13 | [Slides PDF](Slides/Day_6/Slides_QTA_6.pdf) |
| July 7 | [Slides PDF](Slides/Day_2/Slides_QTA_2.pdf) | July 14 | Coming soon |
| July 8 | [Slides PDF](Slides/Day_3/Slides_QTA_3.pdf) | July 15 | Coming soon |
| July 9 | [Slides PDF](Slides/Day_4/Slides_QTA_4.pdf) | July 16 | Coming soon |
| July 10 | [Slides PDF](Slides/Day_5/Slides_QTA_5.pdf) | July 17 | Coming soon |

## Lab Sessions

| Date | Lab | Answers |
| --- | :---: | :---: |
| July 6 | [Quarto file](Lab_sessions/Day_1/Lab_Session_QTA_1.qmd); [data](Lab_sessions/Day_1/hc_sample_1945_2025.rds); [view lab](Lab_sessions/Day_1/Lab_Session_QTA_1.md) | [Quarto file](Lab_sessions/Day_1/Lab_Session_QTA_1_Answers.qmd); [view answers](Lab_sessions/Day_1/Lab_Session_QTA_1_Answers.md) |
| July 7 | [Quarto file](Lab_sessions/Day_2/Lab_Session_QTA_2.qmd); [data](Lab_sessions/Day_2/hc_sample_1945_2025.rds); [view lab](Lab_sessions/Day_2/Lab_Session_QTA_2.md) | [Quarto file](Lab_sessions/Day_2/Lab_Session_QTA_2_Answers.qmd); [view answers](Lab_sessions/Day_2/Lab_Session_QTA_2_Answers.md) |
| July 8 | [Quarto file](Lab_sessions/Day_3/Lab_Session_QTA_3.qmd); [data](Lab_sessions/Day_3/hc_sample_1945_2025.rds); [view lab](Lab_sessions/Day_3/Lab_Session_QTA_3.md) | [Quarto file](Lab_sessions/Day_3/Lab_Session_QTA_3_Answers.qmd); [view answers](Lab_sessions/Day_3/Lab_Session_QTA_3_Answers.md) |
| July 9 | [Quarto file](Lab_sessions/Day_4/Lab_Session_QTA_4.qmd); [data](Lab_sessions/Day_4/hc_sample_1945_2025.rds); [view lab](Lab_sessions/Day_4/Lab_Session_QTA_4.md) | [Quarto file](Lab_sessions/Day_4/Lab_Session_QTA_4_Answers.qmd); [view answers](Lab_sessions/Day_4/Lab_Session_QTA_4_Answers.md) |
| July 10 | [Quarto file](Lab_sessions/Day_5/Lab_Session_QTA_5.qmd); [data](Lab_sessions/Day_5/hc_corbyn_may_party_leaders_2015_2020.rds); [view lab](Lab_sessions/Day_5/Lab_Session_QTA_5.md) | [Quarto file](Lab_sessions/Day_5/Lab_Session_QTA_5_Answers.qmd); [view answers](Lab_sessions/Day_5/Lab_Session_QTA_5_Answers.md) |
| July 13 | [Quarto file](Lab_sessions/Day_6/Lab_Session_QTA_6.qmd); [data](Lab_sessions/Day_6/hc_leader_period_sample_1979_2024.rds); [view lab](Lab_sessions/Day_6/Lab_Session_QTA_6.md) | [Quarto file](Lab_sessions/Day_6/Lab_Session_QTA_6_Answers.qmd); [view answers](Lab_sessions/Day_6/Lab_Session_QTA_6_Answers.md) |
| July 14 | Coming soon | Coming soon |
| July 15 | Coming soon | Coming soon |
| July 16 | Coming soon | Coming soon |
| July 17 | Coming soon | Coming soon |

## Acknowledgements

I thank Stefan Müller for sharing his lab session materials for his QTA course at UCD. I have relied on Codex for streamlining the lab sessions, and for integrating examples from the lab sessions in to the lecture slides.   

## Course schedule


*Day 1 - July 6*

 - **Lecture**: What is quantitative text analysis? And what is quantitative text analysis in the age of LLMs? What will you learn in this course?
 
-  **Lab**: Getting started in Positron and Quarto; basic R objects and data types; loading packages; reading, inspecting, and plotting a sample of House of Commons speeches.

- **Readings**

  - Benoit, K.R. (2020). Text as Data: An Overview. Handbook of Research Methods in Political Science and International Relations. Ed. by L. Curini and R. Franzese. Thousand Oaks: Sage: pp. 461–497.
  - Grimmer, J., Roberts, M.E., and Stewart, B.M. (2022). Text as Data: A New Framework for Machine Learning and the Social Sciences. Princeton: Princeton University Press: chapter 4.

*Day 2 - July 7*

-	**Lecture**: Core assumptions in quantitative text analysis. Representations of text. Preprocessing, feature selection and input design.

-	**Lab**: String operations and regular expressions; cleaning House of Commons metadata; creating and inspecting a `quanteda` corpus, tokens, keywords-in-context, and document-feature matrices.

- **Readings**:

  - Benoit, K., Watanabe, K., Wang, H, Nulty, P., Obeng, A., Müller, & Matsuo, A. (2018). Quanteda: An R package for the quantitative analysis of textual data. Journal of Open Source Software, 3(30), 774.
  - Baden, C., Pipal, C., Schoonvelde, M. & van der Velden, M.A.C.G., (2022). Three Gaps in Computational Text Analysis Methods for Social Sciences: A Research Agenda. Communication Methods and Measures, 16(1): pp. 1–18.

*Day 3 - July 8*

-	**Lecture**: Advanced text representations: embeddings, contextual embeddings, and LLM representations. Comparing representations.

-	**Lab**: Comparing raw text, document-feature matrices, and word embeddings; training a small GloVe model with `text2vec`; inspecting neighbours and cosine similarity.

- **Readings**:

  - Grimmer, J ., Roberts, M.E., and Stewart, B.M. (2022). Text as Data: A New Framework for Machine Learning and the Social Sciences. Princeton: Princeton University Press: chapter 8.
  - Rodriguez, P.L. and Spirling, A., (2022). Word embeddings: What works, what doesn't, and how to tell the difference for applied research. The Journal of Politics, 84(1): pp.101–115.
  
*Day 4 - July 9*

-	**Lecture**: Dictionary construction and validation. How are dictionaries still relevant in the age of LLMs? 

-	**Lab**: Applying sentiment dictionaries; calculating dictionary rates; building, adapting, and validating an economic insecurity dictionary; and using KWIC, embeddings, and LLM prompts to develop candidate terms.

- **Readings**

  - Grimmer, J., Roberts, M.E., and Stewart, B.M. (2022). Text as Data: A New Framework for Machine Learning and the Social Sciences. Princeton: Princeton University Press: chapter 16.
  - Rauh, C., (2018). Validating a sentiment dictionary for German political language–a workbench note. Journal of Information Technology & Politics, 15(4): pp.319–343.

*Day 5 - July 10*

-	**Lecture**: Supervised document classification through human and LLM-assisted coding. Evaluating classifiers and annotations.

-	**Lab**: Defining a supervised classification task; training Naive Bayes and regularised logistic regression classifiers; evaluating predictions with precision, recall, F1, and confusion matrices.

- **Readings**:

  - Daniel Jurafsky and James H. Martin (2026). Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition. 3rd edition: Chapter 4
  - Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks. Proceedings of the National Academy of Sciences of the United States of America 120 (3): e2305016120.

*Day 6 - July 13*

-	**Lecture**: Supervised, semi-supervised, unsupervised, and instruction-based approaches to placing text on an underlying dimension. Discussing their pros and cons.

-	**Lab**: Aggregating speeches into party-period documents; estimating Wordscores, Wordfish, and Latent Semantic Scaling models; comparing what different scaling approaches measure.

- **Readings**:

  - Watanabe, K., (2021). Latent semantic scaling: A semisupervised text analysis technique for new domains and languages. Communication Methods and Measures, 15(2), pp.81-102.
  - Schwemmer, C. and Wieczorek, O., (2020). The methodological divide of sociology: Evidence from two decades of journal publications. Sociology, 54(1): pp.3-21.

*Day 7 - July 14*

-	**Lecture**: Understanding topic models. Discussing their pros and cons. 

-	**Lab**: Estimating LDA, Structural Topic Models, and seeded LDA models; interpreting topic terms, document-topic proportions, and topic prevalence across parties.

- **Readings**:

  - Roberts, M et al. (2014). Structural topic models for open-ended survey responses. American Journal of Political Science, 58(4), pp. 1064–1082.
  - Grimmer, J., Roberts, M.E., and Stewart, B.M. (2022). Text as Data: A New Framework for Machine Learning and the Social Sciences. Princeton: Princeton University Press: chapter 13.

*Day 8 - July 15*

-	**Lecture**: New forms of textual data: multilingual text, speech-to-text, multimodal data. 

-	**Lab**: Using UDPipe for tokenisation, lemmatisation, POS tagging, and dependency parsing; building POS-specific DFMs; comparing parties with keyness, topics, and sentiment.

- **Readings**:

  - Proksch, S.O., Wratil, C. and Wäckerle, J., (2019). Testing the validity of automatic speech recognition for political text analysis. Political Analysis, 27(3), pp. 339-359.
  - Licht, H. and Lind, F., (2023). Going cross-lingual: A guide to multilingual text analysis. Computational Communication Research, 5(2), pp. 1–31. 
  - Birkenmaier, L., Lechner, C.M. and Wagner, C. (2024). The search for solid ground in text as data: A systematic review of validation practices and practical recommendations for validation. Communication Methods and Measures, 18(3), pp. 249-277.

*Day 9 - July 16*

-	**Lecture**: New developments in supervised machine learning. Weak supervision. Transfer learning. 

-	**Lab**: Using Natural Language Inference for zero-shot classification of House of Commons speeches; comparing model labels with weak keyword labels and designing validation workflows.

- **Readings**:

  - Laurer, M., Van Atteveldt, W., Casas, A. & Welbers, K., (2024). Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. Political Analysis, 32(1), pp. 84–100.
  - Kroon, A., Welbers, K., Trilling, D., & van Atteveldt, W. (2024). Advancing automated content analysis for a new era of media eﬀects research: The key role of transfer learning. Communication Methods and Measures, 18(2), pp. 142–162.
  - S. Wankmüller (2024). Introduction to Neural Transfer Learning with Transformers for Social Science Text Analysis. Sociological Methods & Research 53 (4): 1676–1752.
  
*Day 10 - July 17*

-	**Lecture**: LLMs in QTA: annotation, classification, extraction, validation, and reporting. Concluding remarks 

-	**Lab**: LLM-assisted coding with `quallmer` and `ellmer`; writing codebooks, coding House of Commons speeches, comparing prompts or models, validating outputs, and preserving provenance.

- **Readings**:

   - Bail, C.A., (2024). Can Generative AI improve social science? Proceedings of the National Academy of Sciences, 121(21) p.e2314021121.
   - Benoit, K., De Marchi, S., Laver, C., Laver, M. and Ma, J. (2025). Using large language models to analyze political texts through natural language understanding. American Journal of Political Science., pp 1-17.
   - Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2025). On measurement validity and language models: Increasing validity and decreasing bias with instructions. Communication Methods and Measures 19(1), pp 46–62.


