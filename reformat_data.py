import pandas as pd


class ReformatCredit:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def convert_crew_to_data_frame(self, jobs, drop_cols):
        all_crew_dicts = list()
        for movie_id, row in self.data_frame.iterrows():
            crew_list = pd.read_json(row['crew'])
            for crew_member in crew_list.to_dict(orient='records'):
                crew_member['movie_id'] = movie_id
                all_crew_dicts.append(crew_member)
        crew_df = pd.DataFrame(all_crew_dicts)
        crew_df = crew_df[crew_df['job'].isin(jobs)]
        crew_df.drop(columns=drop_cols, inplace=True)
        return crew_df

    def convert_cast_to_data_frame(self, drop_cols):
        all_crew_dicts = list()
        for movie_id, row in self.data_frame.iterrows():
            crew_list = pd.read_json(row['cast'])
            for crew_member in crew_list.to_dict(orient='records'):
                crew_member['movie_id'] = movie_id
                all_crew_dicts.append(crew_member)
        cast_df = pd.DataFrame(all_crew_dicts)
        cast_df.drop(columns=drop_cols, inplace=True)
        return cast_df

    def merge(self):
        crew_df = self.convert_crew_to_data_frame(drop_cols=['id', 'credit_id'],
                                                  jobs=['Director', 'Writer', 'Producer']).drop(columns=['gender'])
        cast_df = (self.convert_cast_to_data_frame(drop_cols=['character', 'credit_id', 'id', 'cast_id'])
                   .rename(columns={'name': 'cast'}))
        self.data_frame.drop(columns=['cast', 'crew'], inplace=True)
        data_frame = pd.merge(left=crew_df, right=cast_df, how='inner', on='movie_id')
        self.data_frame = (pd.merge(left=self.data_frame, right=data_frame, how='inner', on='movie_id')
                           .rename(columns={'movie_id': 'id'}))


class ReformatMovie:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.temp = list()

    def convert_country_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            country_list = pd.read_json(row['production_countries'])
            for country_member in country_list.to_dict(orient='records'):
                country_member['movie_id'] = movie_id
                self.temp.append(country_member)
        country_df = pd.DataFrame(self.temp).rename(columns={'name': 'Country', 'iso_3166_1': 'iso'})
        self.temp = list()
        return country_df

    def convert_company_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            companies_list = pd.read_json(row['production_companies'])
            for company_member in companies_list.to_dict(orient='records'):
                company_member['movie_id'] = movie_id
                self.temp.append(company_member)
        companies_df = pd.DataFrame(self.temp).rename(columns={'name': 'company_name'})
        self.temp = list()
        return companies_df.drop(columns=['id'])

    def convert_genre_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            genres_list = pd.read_json(row['genres'])
            for genre_member in genres_list.to_dict(orient='records'):
                genre_member['movie_id'] = movie_id
                self.temp.append(genre_member)
        genres_df = pd.DataFrame(self.temp).rename(columns={'name': 'genre'})
        self.temp = list()
        return genres_df.drop(columns=['id'])

    def convert_keyword_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            key_words_list = pd.read_json(row['keywords'])
            for key_word_member in key_words_list.to_dict(orient='records'):
                key_word_member['movie_id'] = movie_id
                self.temp.append(key_word_member)
        key_words_df = pd.DataFrame(self.temp).rename(columns={'name': 'key_word'})
        self.temp = list()
        return key_words_df.drop(columns=['id'])

    def convert_spoken_languages_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            spoken_language_list = pd.read_json(row['spoken_languages'])
            for spoken_member in spoken_language_list.to_dict(orient='records'):
                spoken_member['movie_id'] = movie_id
                self.temp.append(spoken_member)
        spoken_languages_df = pd.DataFrame(self.temp).rename(columns={'name': 'key_word', 'iso_639_1': 'code'})
        self.temp = list()
        return spoken_languages_df

    def merge(self):
        companies = self.convert_company_to_data_frame()
        countries = self.convert_country_to_data_frame()
        genres = self.convert_genre_to_data_frame()
        # key_words = self.convert_keyword_to_data_frame()
        # spoken_languages = self.convert_spoken_languages_to_data_frame()
        self.data_frame.drop(columns=['production_companies', 'production_countries', 'genres', 'homepage',
                                      'overview', 'original_title', 'title', 'tagline', 'spoken_languages',
                                      'keywords'], inplace=True)
        data_frame = pd.merge(left=companies, right=countries, how='inner', on='movie_id')
        data_frame = (pd.merge(left=data_frame, right=genres, how='inner', on='movie_id').rename(columns={'movie_id': 'id'}))
        # data_frame = pd.merge(left=data_frame, right=key_words, how='inner', on='movie_id')
        # data_frame = ((pd.merge(left=data_frame, right=spoken_languages, how='inner', on='movie_id'))


                      #
        self.data_frame = pd.merge(left=data_frame, right=self.data_frame, how='inner', on='id')


credit = ReformatCredit(pd.read_csv('tmdb_5000_credits.csv'))
movie = ReformatMovie(pd.read_csv('tmdb_5000_movies.csv'))
movie.merge()
credit.merge()
movie.data_frame.to_csv('tmdb_reformatted_movies.csv', index=False)
credit.data_frame.to_csv('tmdb_reformatted_credits.csv', index=False)