import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

server = app.server


"""
#  Page layout and contents  #	 #	 #	 #	 #	 #	 #	 #	 #	 #	 #
"""

# Load Data
hospital_df = pd.read_csv('all_hospital_reviews_processed.csv')

# Color Pallete
colors = {
	'primary': '#00549f'
}

# Summary Plot
fort_sanders = hospital_df[hospital_df['hospital'] == 'Fort Sanders']
fort_sanders = fort_sanders['review_rating'].value_counts(normalize=True)*100

parkwest = hospital_df[hospital_df['hospital'] == 'Parkwest']
parkwest = parkwest['review_rating'].value_counts(normalize=True)*100

leconte = hospital_df[hospital_df['hospital'] == 'LeConte']
leconte = leconte['review_rating'].value_counts(normalize=True)*100

oakridge = hospital_df[hospital_df['hospital'] == 'Oak Ridge']
oakridge = oakridge['review_rating'].value_counts(normalize=True)*100

morristown = hospital_df[hospital_df['hospital'] == 'Morristown Hamblen']
morristown = morristown['review_rating'].value_counts(normalize=True)*100

roane = hospital_df[hospital_df['hospital'] == 'Roane']
roane = roane['review_rating'].value_counts(normalize=True)*100

claiborne = hospital_df[hospital_df['hospital'] == 'Claiborne']
claiborne = claiborne['review_rating'].value_counts(normalize=True)*100

summary_plot = go.Figure(data=[
	go.Bar(name='Fort Sanders', x=fort_sanders.index, y=fort_sanders[0:4], text=fort_sanders[0:4]),
	go.Bar(name='Parkwest', x=parkwest.index, y=parkwest[0:4], text=parkwest[0:4]),
	go.Bar(name='LeConte', x=leconte.index, y=leconte[0:4], text=leconte[0:4]),
	go.Bar(name='Oak Ridge', x=oakridge.index, y=oakridge[0:4], text=oakridge[0:4]),
	go.Bar(name='Morristown Hamblen', x=morristown.index, y=morristown[0:4], text=morristown[0:4]),
	go.Bar(name='Roane', x=roane.index, y=roane[0:4], text=roane[0:4]),
	go.Bar(name='Claiborne', x=claiborne.index, y=claiborne[0:4], text=claiborne[0:4])
])

summary_plot.update_layout(
	paper_bgcolor='white',
	plot_bgcolor='white'
)

summary_plot.update_traces(texttemplate='%{text:.3s}', textposition='outside', cliponaxis=False)
summary_plot.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')


SUMMARY_PLOT = [
	dbc.CardHeader(html.H5('Percentage of 1 and 5 Star Reviews by Hospital')),
    dcc.Graph(figure=summary_plot)
]

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand('Covenant Google Reviews', className='ml-2')
                    ),
                ],
                align='center',
                no_gutters=True,
            ),
            href='https://www.covenanthealth.com/',
        )
    ],
    color='#00549f',
    dark=True,
    sticky='top',
)

SIDEBAR_TOP_WORD_PLOT = dbc.Jumbotron(
    [
        html.H4(children='Select hospital and review type', className='display-5'),
        html.Hr(className='my-2'),
        html.Label('Select hospital', className='lead'),
        dcc.Dropdown(
            id='top-word-drop',
            clearable=False,
            style={'marginBottom': 50, 'font-size': 12},
            options=[
				{'label': 'All Hospitals', 'value': 'All Hospitals'},
				{'label': 'Fort Sanders', 'value': 'Fort Sanders'},
				{'label': 'Parkwest', 'value': 'Parkwest'},
				{'label': 'LeConte', 'value': 'LeConte'},
				{'label': 'Oak Ridge', 'value': 'Oak Ridge'},
				{'label': 'Morristown Hamblen', 'value': 'Morristown Hamblen'},
				{'label': 'Roane', 'value': 'Roane'},
				{'label': 'Claiborne', 'value': 'Claiborne'}
				],
			value='All Hospitals'
        ),
        html.Label('Select review type', className='lead'),
        dcc.RadioItems(
		    id='top-word-drop-good-bad-radio',
			options=[
				{'label': 'Positive', 'value': 'good'},
				{'label': 'Negative', 'value': 'bad'}
				],
			value='good',
			labelStyle={'display': 'block'}
		)
    ]
)

TOP_WORD_PLOT = [
    dbc.CardHeader(html.H5('Top 10 words used in reviews')),
    dbc.CardBody(
        [
            dcc.Loading(
                id='loading-top-word',
                children=[
                    dbc.Alert(
                        'Not enough data to render this plot, please adjust the filters',
                        id='no-data-alert-top-word',
                        color='warning',
                        style={'display': 'none'},
                    ),
                    dcc.Graph(id='top-words'),
                ],
                color=colors['primary'],
                type='circle',
            )
        ],
        style={'marginTop': 0, 'marginBottom': 0},
    ),
]

SIDEBAR_TOP_BIGRAM_PLOT = dbc.Jumbotron(
    [
        html.H4(children='Select hospital and review type', className='display-5'),
        html.Hr(className='my-2'),
        html.Label('Select hospital', className='lead'),
        dcc.Dropdown(
            id='top-2gram-drop',
            clearable=False,
            style={'marginBottom': 50, 'font-size': 12},
            options=[
				{'label': 'All Hospitals', 'value': 'All Hospitals'},
				{'label': 'Fort Sanders', 'value': 'Fort Sanders'},
				{'label': 'Parkwest', 'value': 'Parkwest'},
				{'label': 'LeConte', 'value': 'LeConte'},
				{'label': 'Oak Ridge', 'value': 'Oak Ridge'},
				{'label': 'Morristown Hamblen', 'value': 'Morristown Hamblen'},
				{'label': 'Roane', 'value': 'Roane'},
				{'label': 'Claiborne', 'value': 'Claiborne'}
				],
			value='All Hospitals'
        ),
        html.Label('Select review type', className='lead'),
        dcc.RadioItems(
		    id='top-2gram-drop-good-bad-radio',
			options=[
				{'label': 'Positive', 'value': 'good'},
				{'label': 'Negative', 'value': 'bad'}
				],
			value='good',
			labelStyle={'display': 'block'}
		)
    ]
)

TOP_BIGRAM_PLOT = [
    dbc.CardHeader(html.H5('Top 10 bi-grams used in reviews')),
    dbc.CardBody(
        [
            dcc.Loading(
                id='loading-top-2gram',
                children=[
                    dbc.Alert(
                        'Not enough data to render this plot, please adjust the filters',
                        id='no-data-alert-top-2gram',
                        color='warning',
                        style={'display': 'none'},
                    ),
                    dcc.Graph(id='top-2grams'),
                ],
                color=colors['primary'],
                type='circle',
            )
        ],
        style={'marginTop': 0, 'marginBottom': 0},
    ),
]

SIDEBAR_TOP_TRIGRAM_PLOT = dbc.Jumbotron(
    [
        html.H4(children='Select hospital and review type', className='display-5'),
        html.Hr(className='my-2'),
        html.Label('Select hospital', className='lead'),
        dcc.Dropdown(
            id='top-3gram-drop',
            clearable=False,
            style={'marginBottom': 50, 'font-size': 12},
            options=[
				{'label': 'All Hospitals', 'value': 'All Hospitals'},
				{'label': 'Fort Sanders', 'value': 'Fort Sanders'},
				{'label': 'Parkwest', 'value': 'Parkwest'},
				{'label': 'LeConte', 'value': 'LeConte'},
				{'label': 'Oak Ridge', 'value': 'Oak Ridge'},
				{'label': 'Morristown Hamblen', 'value': 'Morristown Hamblen'},
				{'label': 'Roane', 'value': 'Roane'},
				{'label': 'Claiborne', 'value': 'Claiborne'}
				],
			value='All Hospitals'
        ),
        html.Label('Select review type', className='lead'),
        dcc.RadioItems(
		    id='top-3gram-drop-good-bad-radio',
			options=[
				{'label': 'Positive', 'value': 'good'},
				{'label': 'Negative', 'value': 'bad'}
				],
			value='good',
			labelStyle={'display': 'block'}
		)
    ]
)

TOP_TRIGRAM_PLOT = [
    dbc.CardHeader(html.H5('Top 10 tri-grams used in reviews')),
    dbc.CardBody(
        [
            dcc.Loading(
                id='loading-top-3gram',
                children=[
                    dbc.Alert(
                        'Not enough data to render this plot, please adjust the filters',
                        id='no-data-alert-top-3gram',
                        color='warning',
                        style={'display': 'none'},
                    ),
                    dcc.Graph(id='top-3grams'),
                ],
                color=colors['primary'],
                type='circle',
            )
        ],
        style={'marginTop': 0, 'marginBottom': 0},
    ),
]


BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(SUMMARY_PLOT)),] , style={'marginTop': 30}),
        dbc.Row(
            [
                dbc.Col(SIDEBAR_TOP_WORD_PLOT, md=4, align='center'),
                dbc.Col(dbc.Card(TOP_WORD_PLOT), md=8),
            ],
            style={'marginTop': 30},
        ),
        dbc.Row(
            [
                dbc.Col(SIDEBAR_TOP_BIGRAM_PLOT, md=4, align='center'),
                dbc.Col(dbc.Card(TOP_BIGRAM_PLOT), md=8),
            ],
            style={'marginTop': 30},
        ),
                dbc.Row(
            [
                dbc.Col(SIDEBAR_TOP_TRIGRAM_PLOT, md=4, align='center'),
                dbc.Col(dbc.Card(TOP_TRIGRAM_PLOT), md=8),
            ],
            style={'marginTop': 30},
        )
    ],
    className='mt-12',
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for Heroku deployment

app.layout = html.Div(children=[NAVBAR, BODY])


"""
#  Callbacks #	#	#	#	#	#	#	#	#	#	#	#
"""

@app.callback(
	Output('top-words', 'figure'),
	[
		Input('top-word-drop', 'value'),
		Input('top-word-drop-good-bad-radio', 'value')
	]
)

# Top Words Plots
def update_top_n_words(hospital, good_bad):
    if hospital == 'All Hospitals':
        data = hospital_df.loc[hospital_df['review_rating'] == good_bad]
        vec = CountVectorizer(stop_words = 'english').fit(data['review_text_processed'])
        bag_of_words = vec.transform(data['review_text_processed'])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq = words_freq[:10]
        common_words_df = pd.DataFrame(words_freq)
        common_words_df.columns=['Word', 'Count']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=common_words_df['Word'],
            y=common_words_df['Count'],
            text=common_words_df['Count'],
            marker_color=colors['primary']
        ))

        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', cliponaxis=False)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_layout({
        	'plot_bgcolor': 'white',
        	'paper_bgcolor': 'white',
        })
    
    else:
        data_hospital = hospital_df.loc[hospital_df['review_rating'] == good_bad]
        data_hospital = data_hospital.loc[data_hospital['hospital'] == hospital]
        vec = CountVectorizer(stop_words = 'english').fit(data_hospital['review_text_processed'])
        bag_of_words = vec.transform(data_hospital['review_text_processed'])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq = words_freq[:10]
        common_words_df = pd.DataFrame(words_freq)
        common_words_df.columns=['Word', 'Count']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=common_words_df['Word'],
            y=common_words_df['Count'],
            text=common_words_df['Count'],
            marker_color=colors['primary']
        ))

        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', cliponaxis=False )
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_layout({
        	'plot_bgcolor': 'white',
        	'paper_bgcolor': 'white',
        })
        
    return(fig)
    
@app.callback(
	Output('top-2grams', 'figure'),
	[
		Input('top-2gram-drop', 'value'),
		Input('top-2gram-drop-good-bad-radio', 'value')
	]
)

    
# Top bi-grams Plots
def update_top_bi_grams(hospital, good_bad):
    if hospital == 'All Hospitals':
        data = hospital_df.loc[hospital_df['review_rating'] == good_bad]
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(data['review_text_processed'])
        bag_of_words = vec.transform(data['review_text_processed'])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq = words_freq[:10]
        ngrams_df = pd.DataFrame(words_freq)
        ngrams_df.columns=['n-gram', 'Count']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=ngrams_df['n-gram'],
            y=ngrams_df['Count'],
            text=ngrams_df['Count'],
            marker_color=colors['primary']
        ))

        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', cliponaxis=False)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_layout({
        	'plot_bgcolor': 'white',
        	'paper_bgcolor': 'white',
        })

    else:
        data_hospital = hospital_df.loc[hospital_df['review_rating'] == good_bad]
        data_hospital = data_hospital.loc[data_hospital['hospital'] == hospital]
        vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(data_hospital['review_text_processed'])
        bag_of_words = vec.transform(data_hospital['review_text_processed'])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq = words_freq[:10]
        ngrams_df = pd.DataFrame(words_freq)
        ngrams_df.columns=['n-gram', 'Count']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=ngrams_df['n-gram'],
            y=ngrams_df['Count'],
            text=ngrams_df['Count'],
            marker_color=colors['primary']
        ))

        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', cliponaxis=False)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_layout({
        	'plot_bgcolor': 'white',
        	'paper_bgcolor': 'white',
        })

    return(fig)
    
@app.callback(
	Output('top-3grams', 'figure'),
	[
		Input('top-3gram-drop', 'value'),
		Input('top-3gram-drop-good-bad-radio', 'value')
	]
)

    
# Top tri-grams Plots
def update_top_tri_grams(hospital, good_bad):
    if hospital == 'All Hospitals':
        data = hospital_df.loc[hospital_df['review_rating'] == good_bad]
        vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(data['review_text_processed'])
        bag_of_words = vec.transform(data['review_text_processed'])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq = words_freq[:10]
        ngrams_df = pd.DataFrame(words_freq)
        ngrams_df.columns=['n-gram', 'Count']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=ngrams_df['n-gram'],
            y=ngrams_df['Count'],
            text=ngrams_df['Count'],
            marker_color=colors['primary']
        ))

        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', cliponaxis=False)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_layout({
        	'plot_bgcolor': 'white',
        	'paper_bgcolor': 'white',
        })

    else:
        data_hospital = hospital_df.loc[hospital_df['review_rating'] == good_bad]
        data_hospital = data_hospital.loc[data_hospital['hospital'] == hospital]
        vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(data_hospital['review_text_processed'])
        bag_of_words = vec.transform(data_hospital['review_text_processed'])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq = words_freq[:10]
        ngrams_df = pd.DataFrame(words_freq)
        ngrams_df.columns=['n-gram', 'Count']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=ngrams_df['n-gram'],
            y=ngrams_df['Count'],
            text=ngrams_df['Count'],
            marker_color=colors['primary']
        ))

        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', cliponaxis=False)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_layout({
        	'plot_bgcolor': 'white',
        	'paper_bgcolor': 'white',
        })

    return(fig)




if __name__ == '__main__':
    app.run_server(debug=True)
