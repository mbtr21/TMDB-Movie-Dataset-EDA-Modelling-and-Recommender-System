import pandas as pd


class ReformatCredit:
    # Constructor to initialize the ReformatCredit object with a pandas DataFrame containing movie credits data.
    def __init__(self, data_frame):
        self.data_frame = data_frame

    # Method to convert 'crew' JSON strings from the DataFrame into a pandas DataFrame,
    # filtering by specified jobs, and dropping specified columns.
    def convert_crew_to_data_frame(self, jobs, drop_cols):
        all_crew_dicts = list()  # List to accumulate crew member dictionaries.
        # Iterates through each movie, extracting its ID and crew data.
        for movie_id, row in self.data_frame.iterrows():
            # Parses the JSON string in 'crew' column to a pandas DataFrame.
            crew_list = pd.read_json(row['crew'])
            # Converts the DataFrame of crew members to dictionaries and appends them to the list with added movie_id.
            for crew_member in crew_list.to_dict(orient='records'):
                crew_member['movie_id'] = movie_id
                all_crew_dicts.append(crew_member)
        # Converts the list of dictionaries to a DataFrame.
        crew_df = pd.DataFrame(all_crew_dicts)
        # Filters the DataFrame for specified jobs.
        crew_df = crew_df[crew_df['job'].isin(jobs)]
        # Drops specified columns.
        crew_df.drop(columns=drop_cols, inplace=True)
        return crew_df

    # Method to convert 'cast' JSON strings from the DataFrame into a pandas DataFrame, dropping specified columns.
    def convert_cast_to_data_frame(self, drop_cols):
        all_crew_dicts = list()  # List to accumulate cast member dictionaries.
        # Iterates through each movie, extracting its ID and cast data.
        for movie_id, row in self.data_frame.iterrows():
            # Parses the JSON string in 'cast' column to a pandas DataFrame.
            crew_list = pd.read_json(row['cast'])
            # Converts the DataFrame of cast members to dictionaries and appends them to the list with added movie_id.
            for crew_member in crew_list.to_dict(orient='records'):
                crew_member['movie_id'] = movie_id
                all_crew_dicts.append(crew_member)
        # Converts the list of dictionaries to a DataFrame.
        cast_df = pd.DataFrame(all_crew_dicts)
        # Drops specified columns.
        cast_df.drop(columns=drop_cols, inplace=True)
        return cast_df

    # Method to merge the transformed crew and cast DataFrames with the original DataFrame,
    # performing necessary transformations like renaming and dropping columns.
    def merge(self):
        # Applies transformations to crew and cast data, including filtering and renaming.
        crew_df = self.convert_crew_to_data_frame(drop_cols=['id', 'credit_id'],
                                                  jobs=['Director', 'Writer', 'Producer']).drop(columns=['gender'])
        cast_df = (self.convert_cast_to_data_frame(drop_cols=['character', 'credit_id', 'id', 'cast_id'])
                   .rename(columns={'name': 'cast'}))
        # Drops 'cast' and 'crew' columns from the original DataFrame.
        self.data_frame.drop(columns=['cast', 'crew'], inplace=True)
        # Merges the transformed crew and cast DataFrames with the original DataFrame on 'movie_id'.
        data_frame = pd.merge(left=crew_df, right=cast_df, how='inner', on='movie_id')
        # Merges the result with the original DataFrame, ensuring all transformations are applied.
        self.data_frame = (pd.merge(left=self.data_frame, right=data_frame, how='inner', on='movie_id')
                           .rename(columns={'movie_id': 'id'}))


class ReformatMovie:
    # Constructor initializes the ReformatMovie object with a pandas DataFrame and an empty list for temporary storage.
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.temp = list()

    # Converts production country JSON data into a pandas DataFrame with specified column names.
    def convert_country_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            country_list = pd.read_json(row['production_countries'])
            # Parse JSON data from 'production_countries' column.
            for country_member in country_list.to_dict(orient='records'):
                country_member['movie_id'] = movie_id  # Add movie ID to each country record.
                self.temp.append(country_member)
        country_df = pd.DataFrame(self.temp).rename(columns={'name': 'Country', 'iso_3166_1': 'iso'})
        # Create and rename DataFrame.
        self.temp = list()  # Clear the temporary list for future use.
        return country_df

    # Converts production company JSON data into a DataFrame, including dropping unnecessary columns.
    def convert_company_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            companies_list = pd.read_json(row['production_companies'])
            for company_member in companies_list.to_dict(orient='records'):
                company_member['movie_id'] = movie_id
                self.temp.append(company_member)
        companies_df = pd.DataFrame(self.temp).rename(columns={'name': 'company_name'})
        self.temp = list()
        return companies_df.drop(columns=['id'])  # Return DataFrame without the 'id' column.

    # Converts genre JSON data into a DataFrame, similar to previous methods.
    def convert_genre_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            genres_list = pd.read_json(row['genres'])
            for genre_member in genres_list.to_dict(orient='records'):
                genre_member['movie_id'] = movie_id
                self.temp.append(genre_member)
        genres_df = pd.DataFrame(self.temp).rename(columns={'name': 'genre'})
        self.temp = list()
        return genres_df.drop(columns=['id'])  # Exclude the 'id' column in the returned DataFrame.

    # Method for converting keywords JSON data into a DataFrame.
    def convert_keyword_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            key_words_list = pd.read_json(row['keywords'])
            for key_word_member in key_words_list.to_dict(orient='records'):
                key_word_member['movie_id'] = movie_id
                self.temp.append(key_word_member)
        key_words_df = pd.DataFrame(self.temp).rename(columns={'name': 'key_word'})
        self.temp = list()
        return key_words_df.drop(columns=['id'])

    # Converts spoken language JSON data into a DataFrame, renaming columns appropriately.
    def convert_spoken_languages_to_data_frame(self):
        for movie_id, row in self.data_frame.iterrows():
            spoken_language_list = pd.read_json(row['spoken_languages'])
            for spoken_member in spoken_language_list.to_dict(orient='records'):
                spoken_member['movie_id'] = movie_id
                self.temp.append(spoken_member)
        spoken_languages_df = pd.DataFrame(self.temp).rename(columns={'name': 'key_word', 'iso_639_1': 'code'})
        self.temp = list()
        return spoken_languages_df

    # Method to merge the converted DataFrames (companies, countries, genres) with the main movie DataFrame.
    def merge(self):
        companies = self.convert_company_to_data_frame()
        countries = self.convert_country_to_data_frame()
        genres = self.convert_genre_to_data_frame()
        # Lines for converting keywords and spoken languages are commented out, indicating optional processing steps.
        self.data_frame.drop(columns=['production_companies', 'production_countries', 'genres', 'homepage',
                                      'tagline', 'spoken_languages', 'original_language',
                                      'keywords', 'original_title'], inplace=True)
        # Drop specified columns from the main DataFrame.
        data_frame = pd.merge(left=companies, right=countries, how='inner', on='movie_id')
        data_frame = (pd.merge(left=data_frame, right=genres, how='inner', on='movie_id')
                      .rename(columns={'movie_id': 'id'}))
        # Merge the transformed data with the main DataFrame on 'id', completing the restructuring.
        self.data_frame = pd.merge(left=data_frame, right=self.data_frame, how='inner', on='id')


credit = ReformatCredit(pd.read_csv('tmdb_5000_credits.csv'))
movie = ReformatMovie(pd.read_csv('tmdb_5000_movies.csv'))
movie.merge()
credit.merge()
iso_alpha_2_codes = ['US', 'GB', 'JM', 'BS', 'DM', 'NZ', 'DE', 'CA', 'FR', 'AU', 'CZ', 'BE', 'IN', 'CN',
                     'JP', 'IT', 'AE', 'ES', 'HK', 'TW', 'MT', 'HU', 'IE', 'ZA', 'MC', 'CH', 'MX', 'FI',
                     'IS', 'DK', 'TN', 'PH', 'LU', 'BA', 'RU', 'BR', 'NL', 'SE', 'KR', 'KZ', 'PK', 'KH',
                     'SK', 'NO', 'AF', 'IL']

# Example mapping for a subset of countries
iso_alpha_2_to_3_complete = {
    'US': 'USA', 'GB': 'GBR', 'JM': 'JAM', 'BS': 'BHS', 'DM': 'DMA', 'NZ': 'NZL', 'DE': 'DEU', 'CA': 'CAN',
    'FR': 'FRA', 'AU': 'AUS', 'CZ': 'CZE', 'BE': 'BEL', 'IN': 'IND', 'CN': 'CHN', 'JP': 'JPN', 'IT': 'ITA',
    'AE': 'ARE', 'ES': 'ESP', 'HK': 'HKG', 'TW': 'TWN', 'MT': 'MLT', 'HU': 'HUN', 'IE': 'IRL', 'ZA': 'ZAF',
    'MC': 'MCO', 'CH': 'CHE', 'MX': 'MEX', 'FI': 'FIN', 'IS': 'ISL', 'DK': 'DNK', 'TN': 'TUN', 'PH': 'PHL',
    'LU': 'LUX', 'BA': 'BIH', 'RU': 'RUS', 'BR': 'BRA', 'NL': 'NLD', 'SE': 'SWE', 'KR': 'KOR', 'KZ': 'KAZ',
    'PK': 'PAK', 'KH': 'KHM', 'SK': 'SVK', 'NO': 'NOR', 'AF': 'AFG', 'IL': 'ISR'
}
# Convert to pandas DataFrame
df_iso_codes = pd.DataFrame(iso_alpha_2_codes, columns=['ISO Alpha-2'])

# Use the mapping to convert Alpha-2 to Alpha-3
movie.data_frame['iso'] = movie.data_frame['iso'].apply(lambda x: iso_alpha_2_to_3_complete.get(x, 'Unknown'))

df_iso_codes.head()
movie.data_frame.to_csv('tmdb_reformatted_movies.csv', index=False)
credit.data_frame.to_csv('tmdb_reformatted_credits.csv', index=False)
