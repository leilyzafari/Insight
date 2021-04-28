# from textblob import TextBlob
import pandas as pd
import joblib

import os
import spacy
nlp = spacy.load('en_core_web_sm')

def get_predicted_dataset(dataset):
	relative_model_path = r"..\model\model_mlb.pkl"
	model_file = os.path.join(os.path.dirname(__file__),relative_model_path)
	model = joblib.load(model_file)

	relative_mlb_path = r"..\model\mlb.pkl"
	mlb_file = os.path.join(os.path.dirname(__file__), relative_mlb_path)
	mlb = joblib.load(mlb_file)

	relative_sentiment_analysis_path = r"..\model\sentiment-analysis.pkl"
	sentiment_analysis_file = os.path.join(os.path.dirname(__file__), relative_sentiment_analysis_path)
	sentiment_analysis_model = joblib.load(sentiment_analysis_file)

	review_list=[]
	sentiment_list=[]
	for index,row in dataset.iterrows():
		text = row['text'].replace('.','. ').replace('!','! ').replace('?','? ')
		for sentence in list(nlp(row['text']).sents):
			if len(sentence.text)>2:
				review_list.append(sentence.text)
			#sentiment_list.append(TextBlob(sentence).sentiment.polarity)
	series=pd.Series(review_list).astype(str)
	predicted = sentiment_analysis_model.predict(series)
	sentiment_series = pd.Series(predicted).astype(str)
	#sentiment_series = pd.Series(sentiment_list).astype(str)
	pred_sentiment_df = pd.DataFrame(
						{'text_pro': series,
						'sentiment':sentiment_series
						})
	predicted = model.predict(pred_sentiment_df['text_pro'])
	pred_df = pd.DataFrame(
			{'text_pro': pred_sentiment_df['text_pro'],
			'sentiment': pred_sentiment_df['sentiment'],
			'pred_category': mlb.inverse_transform(predicted)
			})
	print('head',pred_df.head())
	#pred_df.to_csv("reviews_annotated_version3_clean.csv")
	return pred_df