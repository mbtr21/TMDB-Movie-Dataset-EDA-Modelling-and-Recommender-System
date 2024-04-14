import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Output, Input, State
from tasks import generate_bar_chart, machine_learning
from recommender import similarity


# define layout class for dashboard
class DashboardLayout:
    def __init__(self, app, data_frame):
        self.app = app
        self.data_frame = data_frame
        self._setup_layout()

    # Lay_out of dashboard
    def _setup_layout(self):
        self.app.layout = dbc.Container([
            dcc.Tabs([
                dcc.Tab([
                    dbc.Row([
                        dbc.Col([html.Br(), dcc.Markdown('''
            # Description 
            The below chart is a bar chart which give you ability to compare the items in the drop down of 
            the category which you chose , and from right dropdown you can choose for example name of
            directors and from left dropdown you can choose numerical feature which you want to compare 
            int the bar chart
            ''')])]),
                    dbc.Row([
                        dbc.Col([html.Br(), html.H1('Movie Recommendations', style={'font-size': '20px'})]),
                        dbc.Col([html.Br(), dcc.RadioItems(id="movie-radio-items", options=['Cast', 'Director', 'Genre',
                                                                                            'Producer_Company'],
                                                           value='Director',
                                                           inline=True, className='dbc')])]),
                    dbc.Row([
                        dbc.Col([html.Br(), dcc.Markdown('''
            # Description 
            The below chart is a bar chart which give you ability to compare the items in the drop down of 
            the category which you chose , and from right dropdown you can choose for example name of
            directors and from left dropdown you can choose numerical feature which you want to compare 
            int the bar chart
            ''')])

                    ]),

                    dbc.Row(
                        [
                            dbc.Col([html.Br(), dcc.Dropdown(id='movie-dropdown', className='dbc', multi=True,
                                                             value=list(
                                                                 self.data_frame[self.data_frame['job'] == 'Director'][
                                                                     'name'].
                                                                 unique())[:3])]),
                            dbc.Col([html.Br(), dcc.Dropdown(id='numerical-items', options=['budget', 'popularity',
                                                                                            'vote_average', 'runtime'],
                                                             className='dbc', multi=True, value='budget')], width=3)
                        ]
                    ),
                    dbc.Row(dbc.Col([html.Br(), dcc.Graph(id='bar-chart')])),
                    dbc.Row(dbc.Col([html.Br(), dcc.Markdown(
                        '''
                        # Description 
                        The below chart is a histogram which make you visible to see data from every entity which comes 
                        from every chosen category in a specific data which you can choose from data picker item 
                        '''
                    )])),
                    dbc.Row(
                        [
                            dbc.Col([html.Br(), dcc.Dropdown(id='radio-dropdown')]),
                            dbc.Col([html.Br(), dcc.Dropdown(id='movie-item-dropdown', options=['budget', 'popularity',
                                                                                                'vote_average',
                                                                                                'runtime',
                                                                                                'revenue',
                                                                                                'vote_count'],
                                                             value='budget',
                                                             className='dbc')]),
                            dbc.Col([html.Br(), dcc.DatePickerRange(id='date-picker',
                                                                    min_date_allowed=self.data_frame[
                                                                        'release_date'].min(),
                                                                    max_date_allowed=self.data_frame[
                                                                        'release_date'].max(),
                                                                    initial_visible_month=self.data_frame[
                                                                        'release_date'].max(),
                                                                    start_date=self.data_frame['release_date'].min(),
                                                                    end_date=self.data_frame['release_date'].max(),
                                                                    display_format='YYYY-MM-DD')])
                        ]
                    ),
                    dbc.Row([dbc.Col([html.Br(), dcc.Graph(id='line-chart')])
                             ]),
                    dbc.Row(
                        [
                            dbc.Col(
                                [html.Br(), dcc.Graph(figure=px.pie(data_frame=self.data_frame, names='Genre',
                                                                    values='budget',
                                                                    title='Budget over genres', hole=.7,
                                                                    color_discrete_sequence=px.colors.sequential.
                                                                    Darkmint)), ]),

                            dbc.Col([html.Br(), dcc.Graph(figure=px.pie(self.data_frame, values='popularity',
                                                                        names='Genre',
                                                                        title='Popularity over Genres', hole=.7,
                                                                        color_discrete_sequence=px.colors.sequential.
                                                                        Darkmint))])
                        ]
                    ),
                    dbc.Row([
                        dbc.Col([html.Br(), dcc.Markdown('''
        # Description 
        The follow heat map is a heat map for correlation between each numerical data columns in data
        and you can choose the metric from radio items for calculate the correlation  
        ''')])]),
                    dbc.Row([
                        dbc.Col(
                            [
                                html.Br(),
                                dcc.RadioItems(id='Radio-heat-map', options=['pearson', 'kendall', 'spearman'],
                                               value='pearson')
                            ]
                        ),
                        dbc.Col([html.Br(), dcc.Graph(id='heat-map')])
                        ]
                    ),
                    dbc.Row([
                        dbc.Col([html.Br(), dcc.Markdown('''
                        # Description 
                        The bellow map is can be use for see distribution of every items in the following dropdown,
                        int the each country
                        ''')])
                    ]),
                    dbc.Row([
                        dbc.Col([html.Br(), dcc.Dropdown(id='map-dropdown', options=['budget', 'popularity',
                                                                                     'vote_average', 'runtime',
                                                                                     'revenue',
                                                                                     'vote_count'], value='budget',
                                                         className='dbc')]),
                        dbc.Col(
                            [html.Br(),
                             dcc.RadioItems(id='map-radio', options=['sum', 'mean', 'min', 'max', 'std'], value='mean',
                                            inline=True)])
                    ]),
                    dbc.Row(
                        [
                            dbc.Col([html.Br(), dcc.Graph(id='map')])
                        ]
                    ),
                ], label='EDA_Dashboard'),
                dcc.Tab(
                    [
                        dbc.Row([
                            dbc.Col([html.Br(), dcc.Markdown('''
                            # Description 
                            In the following dashboard you can choose some items  as feature for re learning models
                            like numerical columns to convert to log of them or some categorical columns for 
                            one hot encoding after that you can compare the result metrics for each model in the
                            bar chart and we set a default result for this
                            ''')])
                        ]),
                        dbc.Row([
                            dbc.Col([html.Br(), html.H3('select columns for log ')]),
                            dbc.Col([html.Br(), html.H3('select columns for one hot encoding')])
                        ]),
                        dbc.Row([
                            dbc.Col([dcc.Dropdown(id='log-dropdown',
                                                  options=self.data_frame.select_dtypes(
                                                      include='number').columns,
                                                  multi=True,
                                                  # value=self.data_frame.select_dtypes(include='number').columns[0]
                                                  )]),
                            dbc.Col([dcc.Dropdown(id='One-hot-dropdown',
                                                  options=self.data_frame.select_dtypes(
                                                      include='object').columns,
                                                  multi=True, value='Genre')])

                        ]),
                        dbc.Row([
                            dbc.Col([html.Br(), dcc.RadioItems(id='classifier-items', options=['k_means', 'dbscan'],
                                                               inline=True, value='k_means')]),
                            dbc.Col(
                                [html.Br(), dcc.RadioItems(id='dimension-reduction', options=['PCA'], value='PCA')]),
                            dbc.Col([html.Br(),
                                     dcc.RadioItems(id='number-of-features-mutual-info', options=['0.3', '0.5', '0.7'],
                                                    inline=True, value='0.3')])
                        ]),
                        dbc.Row([
                            dbc.Col([html.Br(),
                                     dcc.Dropdown(id='metrics-dropdown', options=['Precision', 'Recall', 'Accuracy',
                                                                                  'F1_score'],
                                                  multi=True, value='F1_score'),
                                     dbc.Col([html.Br(), dcc.Graph(id='Ml-barchart')])
                                     ])
                        ]
                        ),
                        dbc.Row(
                          dbc.Col([html.Br(), dcc.Markdown(
                              ''' 
                              # Description
                              You can enter the name of movie which you want in the follow blank then get the 
                              recommends   
                              ''')])

                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [html.Br(), dcc.Input(id='input-text', type='text',
                                                          placeholder='Enter the name of movie'),
                                     html.Div(id='container')]),
                                dbc.Col([html.Br(), html.Button('Submit', id='submit-btn', n_clicks=0)]),
                                dbc.Col([html.Br(), html.Div(id='output-container')])
                            ]
                        )
                    ], label='Machine Learning Dashboard and recommendation'
                )])], style={'marginTop': 50, 'marginBottom': 20, 'marginLeft': 20, 'marginRight': 20})


# Registering all call back functions for dashh app
class CallbackManager:
    def __init__(self, app, data_frame, machine_learning_data_frame, target, recommender_data):
        self.app = app
        self.data_frame = data_frame
        self.recommender_data = recommender_data
        self.target = target
        self.machine_learning_data_frame = machine_learning_data_frame
        self._register_callbacks()

    def _register_callbacks(self):
        @self.app.callback(
            Output('output-container', 'children'),
            [Input('submit-btn', 'n_clicks')],
            [State('input-text', 'value')]
        )
        def recommender_text(n_clicks, input_value):
            if n_clicks > 0 and input_value is not None:
                df = similarity(self.recommender_data, input_value)
                return html.Table([
                    html.Thead(html.Tr([html.Th(df.name)])),
                    html.Tbody([
                        html.Tr([html.Td(value)]) for value in df
                    ])
                ])

        @self.app.callback(
            Output('Ml-barchart', 'figure'),
            Input('log-dropdown', 'value'),
            Input('metrics-dropdown', 'value'),
            Input('classifier-items', 'value')

        )
        def machine_learning_figure(log_dropdown, metrics_dropdown, classifier_items):
            if log_dropdown is None:
                fig = px.bar(data_frame=self.machine_learning_data_frame, x='name', y=metrics_dropdown,
                             color_discrete_sequence=px.colors.sequential.Darkmint)
            else:
                df = machine_learning(self.data_frame, log_dropdown, self.target, classifier_items)
                fig = px.bar(data_frame=df, x='name', y=metrics_dropdown,
                             color_discrete_sequence=px.colors.sequential.Darkmint)

            return fig

        @self.app.callback(
            Output('movie-dropdown', 'options'),
            Output('radio-dropdown', 'options'),
            Input('movie-radio-items', 'value')
        )
        def set_drop_down(choose):
            list()
            if choose == 'Director':
                options = list(self.data_frame[self.data_frame['job'] == 'Director']['name'].unique())
            else:
                options = list(self.data_frame[f'{choose}'].unique())
            return options, options

        @self.app.callback(
            Output('bar-chart', 'figure'),
            Input('movie-radio-items', 'value'),
            Input('movie-dropdown', 'value'),
            Input('numerical-items', 'value')
        )
        def bar_chart(radio_item, movie_item, numerical_item):
            if radio_item == 'Director':
                df = self.data_frame[self.data_frame['job'] == 'Director']
                df = df[df['name'].isin(movie_item)]
                df = df.groupby('name').agg({'budget': 'mean', 'popularity': 'mean', 'vote_average': 'mean',
                                             'runtime': 'mean', 'revenue': 'mean', 'vote_count': 'mean'})
                df.reset_index(inplace=True)
                fig = px.bar(
                    data_frame=df,
                    x='name',
                    y=numerical_item,
                    color_discrete_sequence=px.colors.sequential.Darkmint

                )
            else:
                df = generate_bar_chart(self.data_frame, radio_item, movie_item)
                df.reset_index(inplace=True)
                fig = px.bar(
                    data_frame=df,
                    x=radio_item,
                    y=numerical_item,
                    color_discrete_sequence=px.colors.sequential.Darkmint
                )
            return fig

        @self.app.callback(
            Output('line-chart', 'figure'),
            Input('movie-radio-items', 'value'),
            Input('radio-dropdown', 'value'),
            Input('movie-item-dropdown', 'value'),
            Input('date-picker', 'start_date'),
            Input('date-picker', 'end_date')
        )
        def hist_chart(radio_item, movie_item, movie_item_2, start_date, end_date):
            df = self.data_frame.loc[self.data_frame['release_date'].between(start_date, end_date)]
            if radio_item == 'Director':
                df = df[df['name'] == movie_item]
            else:
                df = df[df[radio_item] == movie_item]
            fig = px.histogram(data_frame=df, x=movie_item_2, color_discrete_sequence=px.colors.sequential.Darkmint)
            return fig

        @self.app.callback(
            Output('map', 'figure'),
            Input('map-dropdown', 'value'),
            Input('map-radio', 'value')
        )
        def map_generate(drop_down, method):
            df = (self.data_frame.groupby('iso')
                  .agg({'budget': f'{method}', 'popularity': f'{method}', 'vote_average': f'{method}',
                        'revenue': f'{method}', 'vote_count': f'{method}'}))
            df.reset_index(inplace=True)
            figure = px.choropleth(
                data_frame=df,
                locations='iso',
                color=drop_down,
                color_continuous_scale=px.colors.sequential.Darkmint
            )
            figure.update_layout(width=None, height=600)
            return figure

        @self.app.callback(
            Output('heat-map', 'figure'),
            Input('Radio-heat-map', 'value')
        )
        def heat_map(method):
            figure = px.imshow(
                (self.data_frame.select_dtypes(include=['number']).drop(columns=['gender']).corr(method=method)),
                x=(self.data_frame.select_dtypes(include=['number']).drop(columns=['gender'])).columns,
                y=(self.data_frame.select_dtypes(include=['number']).drop(columns=['gender'])).columns,
                labels=dict(color='Correlation'),
                color_continuous_scale=px.colors.sequential.Darkmint
            )
            return figure
