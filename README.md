# Twitter Sentiment Analysis Project using scikit-learn

Tweets are short (140 character or less) strings that often convey strong opinions or emotions. During the 2012 presidential election, a large number of tweets about the presidential debate were collected. In this report, we detail how we created classifiers to determine the sentiment of the tweets with regards to a certain presidential candidate.

Introduction
============

The goal of this project was to classify tweets from the United States 2012 presidential election. More formally, given a set of tweets (strings) that are labeled with a presidential candidate (Obama or Romney) and a rating, y in \{-1,0,1\}, as training data, we were asked to create a classifier that could predict the rating, y, for a new tweet that was only labeled by presidential candidate. To determine the \`\`best" classifier we compared F-score, precision, recall, and overall accuracy for several different classifiers, as well as an ensemble of classifiers. We believe that F-score of the positive and negative class and overall accuracy were the best indicators of a useful classifier, so we sought to build classifiers that maximized these values. For each trial run, we used 80% of the data as training data and 20% of the data as testing data. To analyze each classifier, we performed 10-fold cross validation on the data provided. For this project we primarily utilized the Sci-Kit Learn (http://scikit-learn.org/) library for python. We additionally tried the TensorFlow (https://www.tensorflow.org/) library but focused our work on the Sci-Kit library.

Preprocessing
=============

The first major step towards creating any classifier is performing some preprocessing steps to clean the data. We performed the following preprocessing steps:

-   <span>Removed HTML and Twitter tags</span>

-   Converted words to lower case

-   Removed common stop words that convey no sentiment or opinion. Some examples of stop words are \`\`and“, \`\`the”, and \`\`it". For a complete list of stop words that we removed, please see the appendix.

-   Removed empty spacing, punctuation, special characters and tabs

-   Replaced words that had repeated letters that led to incorrect spellings. For example, if we found a string \`\`loooooooooong“, we replaced this with the word \`\`long”. One problem that this posed was when a word could be shortened by deleting a repeated letter, but there are two possible shorter words that could be created. For example, if we encountered the string \`\`weeeeeeeee“, it is unclear without context if this word should be replaced with \`\`we” or \`\`wee". Whenever treating the tweet as a bag of words, context is essentially lost, so this issue is difficult to resolve.

-   Removed numbers, such as \`\`65“ and \`\`fourty-five” from the tweets, as they convey no apparent sentiment or opinion

-   Corrected spelling for words that ended in a \`\`z“ instead of an \`\`s” since, presumably, the \`\`z“ at the end is meant to pluralize the stem of the word, but causes classifiers to think that it is a new word. For example, the word \`\`presidentz” is presumably the plural of \`\`president“, so we replaced \`\`presidentz” with \`\`presidents".

-   Implemented a stemming process to replace words with their stem word. For example, the words \`\`engineering“ would be replaced with \`\`engineer”. This helps to reduce the size of the vocabulary used and to increase the density of the words

We found that preprocessing played a huge part in increasing the statistics (accuracy, f-score, recall, and precision) of our classifiers, though some steps seemed to affect the classifiers more than others. To see the effect of some of these steps, we plotted the scores of the classifiers with and without certain of the preprocessing steps listed above. It was interesting to see that stemming actually seemed to create worse classifiers in several cases. Additionally, we found that looking at phrases as 1-, 2-, and 3-grams had significant effect on the classifiers with respect to some of the metrics, shown in figure 1. However, there does not seem to be a clear indication, without choosing a single metric to focus on, which value of $n$ is best for $n$-grams.

Supervised Classifiers
======================

We focused our efforts on making four classifiers: SVM, Logistic Regression, Naive Bayes, and an ensemble method classifier. Each of these has a set of parameters that we spent time tweaking in order to improve our results. For example, we compared the value of $\alpha$ in the SVM and found that setting $\alpha = .0001$ seemed to give the best results, as displayed by the graph in figure 2. We also tested classifiers using one sentiment lexicon downloaded from Dr. Bing Liu’s website (https://www.cs.uic.edu/ liub/FBS/sentiment-analysis.html) and one lexicon from the University of Pittsberg (http://mpqa.cs.pitt.edu/lexicons/), however, we did not see much improvement to the scores, and in some cases, noticeable decreases in performance.

![The effect of using 1-, 2-, and 3-grams on four
classifiers.](NGram.png)

![Values of alpha affecting the output of our SVM Classifier. Note that
the x-axis in the $-log$ scale.](SVMAlphaEffect.png)

In the end, our best supervised classifier in terms of accuracy and F score ended up being the SVM for Obama, which gave the most consistent results for these values. Our results for all four classifiers are in figures 3 and 4.

![Obama results.](ObamaSupervised.png)

![Romney Results.](RomneySupervised.png)

Extension: Semi-Supervised Classifiers
======================================

In supervised learning, the labels on all training data are known. In semi-supervised learning, classifiers may be built using some labeled examples and some unlabeled examples. The semi-supervised learning occurs as the unlabeled data are given labels by a base classifier, and then those data are considered to be new, labeled training data, and the classifier is updated based on this new training data.

In the previous section all of the classifiers performed supervised learning on tweets that were labeled with a 0, 1, or -1. However, the original data given to us contained tweets that were given a score of 2. For this project, we were instructed not to use these tweets in the training data, as we only needed to consider the positive, negative, and
neutral class. We thought that an interesting extension would be to try to apply semi-supervised learning, using the tweets that were originally labeled as 2 (and thus not in any of the training sets in the supervised classifiers) as unlabeled training data. The results indicate that using semi-supervised learning to create more training data improves the classifiers across the board, but just slightly. We report our findings with the following graph of, which compares the accuracy, F scores, precision, and recall of our classifiers using semi-supervised and supervised learning.

![Results for Obama tweets using semi-supervised
learning.](ObamaSemisupervised.png)

![Results for Romney tweets using semi-supervised
learning.](RomneySemisupervised.png)

Conclusion
==========

We found this project to be a worthwhile study of the effect of parameters and model choice for sentiment analysis of tweets. It additionally gave us experience researching the best tools for the job, and getting hands on experience using these tools. In the end, we were able to achieve decent classification with the four classifiers we focused on. Our conclusion is that there is no magic bullet for creating classifiers for these such tasks, rather many attempts, parameters, and different techniques must be tried in order to achieve acceptable results. Moreover, there is no one single \`\`best" way to create these classifier, but experience, patience, and critical thinking are key to improvement.

# About Project
This project was done as a part of CS 583: Data Mining and Text Mining at UIC.