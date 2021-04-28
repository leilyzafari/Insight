# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request, redirect
from scrape import scrape
# Create the application object
import sys
import argparse
# import barchart_test
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import model
import pandas as pd
# from sklearn.externals import joblib
# import nltk

from collections import defaultdict

review_dict = defaultdict(list)

# pred_df = pd.read_csv('reviews_annotated_version3_clean.csv')

valueList = []
valueDict = {}
categories = []
numbers = []


def calculate_bar_chart(predicted_df):
    for index, row in predicted_df.iterrows():
        if row['sentiment'] == 'negative':
            # if float(row['sentiment']) < -0.2:
            # print (row['pred_category'])
            for values in row['pred_category']:
                value = values.replace("'", '').replace("'", '').replace('"', '').replace('(', '').replace(')',
                                                                                                           '').strip()
                if len(value) >= 2:
                    valueList.append(value)
                    review = row['text_pro']
                    review_dict[value].append(review)
    for value in set(valueList):
        # if value!='cleanliness':
        valueDict[value] = valueList.count(value)
    categories.extend(list(valueDict.keys()))
    numbers.extend(list(valueDict.values()))
    print("review_dict", review_dict)


app = Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=app, url_base_pathname='/dash_app/')

colors = {
    'background': 'white',
    'text': 'black'
}

dash_app.layout = html.Div([
    html.Div([
        html.Div([
            html.H3('Areas to improve'),
            dcc.Graph(
                id='bar_chart',
                figure={
                    'data': [
                        {'x': numbers, 'y': categories, 'type': 'bar', 'orientation': 'h'},
                    ],
                    'layout': {
                        'xaxis': {
                            'title': 'Number of sentences'
                        },
                        'yaxis': {
                            'title': 'Categories'
                        },
                        'clickmode': 'event',
                        'autosize': False,
                        'width': 600,
                        'height': 500,
                        'plot_bgcolor': colors['background'],
                        'paper_bgcolor': colors['background'],
                        'font': {
                            'color': colors['text']
                        },
                        'margin': {
                            'l': 180,
                        }
                    }
                }
            )
        ], className="six columns"),

        html.Div([
            html.H3('Samples of comments'),
            html.H1(id='my-div-space1'),
            html.H1(id='my-div-space2'),
            html.H1(id='my-div-space3'),
            html.H1(id='my-div-space4'),
            html.H1(id='my-div-space5'),
            html.Div(id='my-div-space6'),
            html.Div(id='my-div1'),
            html.Div(id='my-div-space7'),
            html.Div(id='my-div2'),
            html.Div(id='my-div-space8'),
            html.Div(id='my-div3'),
        ], className="Examples"),
    ])
])


@dash_app.callback(
    [dash.dependencies.Output(component_id='my-div1', component_property='children'),
     dash.dependencies.Output(component_id='my-div2', component_property='children'),
     dash.dependencies.Output(component_id='my-div3', component_property='children')],
    [dash.dependencies.Input(component_id='bar_chart', component_property='clickData')]
)
def update_output_div(clickData):
    print("clickData",clickData)
    if clickData:
        category = clickData['points'][0]['y']
        reviewList = review_dict[category]
        print("reviewList", reviewList)
        if len(reviewList) == 1:
            return [reviewList[0], '', '']
        elif len(reviewList) == 2:
            return [reviewList[0], reviewList[1], '']
        elif not len(reviewList):
            return ['','','']
        return [reviewList[2], reviewList[0], reviewList[1]]
    else:
        return ['','','']


@app.route('/', methods=["GET", "POST"])
def home_page():
    return render_template('index.html')  # render a template


@app.route('/output')
def tag_output():
    #
    # Pull input
    url_input = request.args.get('user_input')
    # Case if empty
    if url_input == '':
        dataset = pd.read_csv('reviews_cora_lunch.csv', delimiter=',')
        predicted_df = model.get_predicted_dataset(dataset)
        calculate_bar_chart(predicted_df)
        return redirect('/dash_app')  # render_template("index.html",my_input = some_input,my_form_result="Empty")
    else:
        # some_image="giphy.gif"
        print(dir(scrape))
        scraper = scrape.TripadvisorScraper()
        sys.argv = ['']
        parser = argparse.ArgumentParser(description='Scrape restaurant reviews from Tripadvisor (.com or .de).')
        parser.add_argument('-url', '--url',
                            default="https://www.tripadvisor.com/Restaurant_Review-g155019-d1308932-Reviews-Blu_Ristorante-Toronto_Ontario.html",
                            help='URL to a Tripadvisor restaurant page')
        parser.add_argument('-o', '--out', dest='outfile', help='Path for output CSV file', default='reviews.csv')
        parser.add_argument('-n', dest='max', help='Maximum number of reviews to fetch', default=sys.maxsize, type=int)
        parser.add_argument('-e', '--engine', dest='engine', help='Driver to use',
                            choices=['phantomjs', 'chrome', 'firefox'], default='chrome')
        args = parser.parse_args()
        dataset = scraper.fetch_reviews(url_input)
        predicted_df = model.get_predicted_dataset(dataset)
        calculate_bar_chart(predicted_df)
        return redirect('/dash_app')


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True)  # will run locally http://127.0.0.1:5000/
# barchart_test.main()
