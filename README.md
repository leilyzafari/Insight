# ReviewSmart
Online reviews play a great role for business. Studies have shown that 94% of people let themselves be convinced to avoid a restaurant by online reviews. This means that restaurants should be aware of any negative comments about them on the internet.

## Context and Use case
It is shown that one third of people search deliberately for what other clients have not liked or have found unpleasant when reading reviews. Interestingly, lots of people mention their criticism in highly ranked reviews, i.e. they are overall happy, but still find topics to be improved. This means in order to find criticized points it is not enough to analyze reviews with only one or two stars, all reviews should be equally analyzed and hidden complaints should be extracted. This could be very time-consuming if done manually.
To facilitate this task, I have implemented ReviewSmart a web app that applies multi-class/multi-label classification to point out client complaints and their categories.

## Data Collection
For collecting data Tripadvisor.com is scraped.
I have used python BeautifulSoup and Selenium for web-scraping and data collection.
This data which consists of client reviews, is split into individual sentences using spacy.

## Model Training
Two models are trained for this product:
* A multi-class Naive Bayes classifier for sentiment analysis
* A multi-label Naive Bayes classifier for assigning sentences to different categories/ labels. Each sentence can belong to multiple labels.

The train set for sentiment analysis should have train data for positive, negative as well as neutral sentences. The reviews could consist of sentences such as "I went there for my birthday", which does not contain any positive or negative sentiment. I could not find any dataset of restaurant reviews that was already labeled with positive, negative and neutral sentiment. I manually labeled a dataset or in other words compiled my trainset myself, trying my best that there would be no data leakage between the train test and the use case data.

## Visualization
The output of the two models will contain negative sentences and their labels (food, service, price, and ambiance).
The result is shown as an interactive bar chart showing the number of negative sentences of each category. When the user clicks on each bar, three of the sentences in that category will be shown.

# Deployment
The product can be used in a web app I have implemented using flask.

# Result
As a result, restaurant owners can see very easily where they should improve and also read the complaints in detail.
As an extension, complaints could be linked back to the original reviews, so by clicking on them you could get back to the user and give a comment on their review.
This in turn will have a positive impression on clients.

