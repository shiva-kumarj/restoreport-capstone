import psycopg2
import pandas as pd
import numpy as np
import os
import logging
import argparse
from sqlalchemy import create_engine

# pd.set_option('future.no_silent_downcasting', True)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Clean Yelp Business dataset and Ingest it into Postgres")
    # user,
    parser.add_argument('--user', help='user name for postgres') 
    # password, 
    parser.add_argument('--password', help='password for postgres') 
    # host
    parser.add_argument('--host', help='host for postgres')
    # port
    parser.add_argument('--port', help='port for postgres') 
    # database
    parser.add_argument('--db', help='db name for postgres') 
    # table
    parser.add_argument('--table_name', help='table name for postgres')
    # url
    parser.add_argument('--url', help='csv url for postgres') 

    # parser.add_argument('--input', dest='input_file', required=True, help='Path to the input JSON file')
    # parser.add_argument('--output', dest='output_file', required=True, help='Path to the output JSON file')
    return parser.parse_args()

def impute_hours(row, default_hours):
    """
    Impute missing hours in the row with the default_hours.

    Parameters:
    - row: DataFrame row containing 'hours' information.
    - default_hours: Default hours to use for imputation.

    Returns:
    Imputed hours.
    """
    if pd.isnull(row):
        return default_hours
    else:
        return row

# Define a function to clean the values
def clean_category(value):
    """
    Clean and preprocess a string value.

    This function removes leading characters and checks for the string "None" to convert it to None.

    Parameters:
    - value (str): The input string to be cleaned.

    Returns:
    - str or None: The cleaned string value. If the input is "None", returns None.
    """
    if value:
        value = value.strip("'u")
        if value == "None":
            return None
        else:
            return value
    return value

def ingest_into_db(df, args):
    user = args.user
    password = args.password
    host = args.host
    port = int(args.port)
    db = args.db
    table_name = args.table_name
    logging.info('Trying to establish connection with Db..')
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}')
    df.head(n=0).to_sql(name=f'{table_name}', con=engine, if_exists='replace')
    logging.info('Ingesting dataframe into Database...')
    df.to_sql(name=f'{table_name}', con=engine, if_exists='append')

def successfully_ingested(df, args):
    user = args.user
    db = args.db
    password = args.password
    host = args.host

    # Create a connection to the database
    conn = psycopg2.connect(host=f'{host}', database=f'{db}', user=f'{user}', password=f'{password}')

    # Create a cursor
    cursor = conn.cursor()

    # Execute an SQL statement
    cursor.execute('SELECT count(*) FROM business')

    # Fetch the results
    results = cursor.fetchall()

    dataframe_shape = df.shape
    # print(dataframe_shape)


    # Close the cursor
    cursor.close()

    # Close the connection
    conn.close()
    
    # Print the results
    if results[0][0] == dataframe_shape[0]:
        return True

    return False


def parse_json_string(json_string, default):
    """
    Parse a JSON string and return the key associated with the first True value.

    This function evaluates the input JSON string, iterates through its key-value pairs,
    and returns the key of the first pair where the value is True. If no True value is found,
    it returns the specified default value.

    Parameters:
    - json_string (str): The JSON string to be parsed.
    - default: The default value to return if no True value is found.

    Returns:
    - Any: The key associated with the first True value or the default value.
    """
    json_string = eval(json_string)
    for k,v in json_string.items():
        if v == True:
            return k
    return default

def clean_business_dataset(input_file):
    
    df = pd.read_json(input_file, encoding='utf-8', lines=True)

    # Considering all USA states
    usa_states = ('AL', 'KY', 'OH', 'AK', 'LA', 'OK', 'AZ', 'ME', 'OR', \
              'AR', 'MD', 'PA', 'AS', 'MA', 'PR', 'CA', 'MI', 'RI', \
              'CO', 'MN', 'SC', 'CT', 'MS', 'SD', 'DE', 'MO', 'TN', \
              'DC', 'MT', 'TX', 'FL', 'NE', 'TT', 'GA', 'NV', 'UT', \
              'GU', 'NH', 'VT', 'HI', 'NJ', 'VA', 'ID', 'NM', 'VI', \
              'IL', 'NY', 'WA', 'IN', 'NC', 'WV', 'IA', 'ND', 'WI', \
              'KS', 'MP', 'WY')

    # ### Cleaning
    
    logging.info('Beginning Cleaning steps...')

    df = df[df['categories'].notnull()]

    # filter businesses Open only in PA and FL.
    # state_filter = ((df['state'] == 'PA') | (df['state'] == 'FL'))
    logging.info(f'Filtering \'Restaurants\' from the dataset')
    usa_filter = (df['state'].isin(usa_states))

    is_open = (df['is_open'] == 1)

    df = df[usa_filter & is_open]

    # make all categories lower case
    df.loc[:, 'categories'] = df['categories'].str.lower()

    # get "restaurants"
    df = df[df['categories'].str.contains('restaurants')]


    logging.info('Fixing \'Postal Code\' ')
    # Postal code
    # assigning placeholder postal codes 
    df.loc[df['postal_code'].apply(lambda x: len(x)<5), 'postal_code'] = 99999
    # Convert to suitable dtype 
    # Bug
    df.loc[:, 'postal_code'] = df['postal_code'].astype('int')


    # Hours
    logging.info('Fixing \'Hours\' ')
    default_hours = {'Monday': None,
                    'Tuesday': None,
                    'Wednesday': None,
                    'Thursday': None,
                    'Friday': None,
                    'Saturday': None,
                    'Sunday': None}

        
    df.loc[:, 'hours'] = df['hours'].apply(impute_hours, default_hours=default_hours)

    days_of_week = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
    for day in days_of_week:
        df.loc[:, f'{day}'] = df['hours'].apply(lambda x: x.get(f'{day}', None))

    for day in days_of_week:
        not_null = df[day].notnull()
        mean_opening = df[not_null][day].apply(lambda x: int(x.split('-')[0].split(':')[0])).mean()
        mean_closing = df[not_null][day].apply(lambda x: int(x.split('-')[1].split(':')[1])).mean()
        df.loc[df[day].isnull(), day] = f'{int(mean_opening)}:00 - {int(mean_closing)}:00'


    # Attributes
    logging.info('Fixing \'Attributes\' column')

    # creating a set of all attributes.
    df = df[df['attributes'].notnull()]
    all_attributes = set()
    df['attributes'].apply(lambda x: all_attributes.update(x.keys()))

    # Create separate columns for each attribute.
    for attribute in all_attributes:
        df.loc[:, f'attributes.{attribute}'] = None

    # map each attribute with its corresponding value from the 'attribute' column.
    for attribute in all_attributes:
        df.loc[:, f'attributes.{attribute}'] = df['attributes'].apply(lambda x: x.get(f'{attribute}', None))

    logging.info('Imputing values in all binary columns')
    # Defining all binary attributes
    binary_attributes = ['attributes.GoodForKids', 'attributes.RestaurantsGoodForGroups', 'attributes.BikeParking', 'attributes.RestaurantsReservations',
                        'attributes.HasTV', 'attributes.Caters', 'attributes.OutdoorSeating', 'attributes.WheelchairAccessible', 'attributes.RestaurantsDelivery',
                        'attributes.RestaurantsTakeOut', 'attributes.BusinessAcceptsCreditCards']

    # Fill all missing values in binary columns with a False
    df.loc[:, binary_attributes] = df[binary_attributes].fillna(False)

    # correcting existing values in the dataframe
    # df.loc[:, binary_attributes] = df[binary_attributes].replace('None', False)
    # df.loc[:, binary_attributes] = df[binary_attributes].replace('True', True)
    # df.loc[:, binary_attributes] = df[binary_attributes].replace('False', False)

    for col in binary_attributes:
        df.loc[df[col] == 'None', col] = False
        df.loc[df[col] == 'True', col] = True
        df.loc[df[col] == 'False', col] = False


    logging.info('Dropping redundant columns')
    # Dropping redundant columns
    drop_columns = ['attributes.RestaurantsAttire', 'attributes.CoatCheck', 'attributes.ByAppointmentOnly', 
                    'attributes.DogsAllowed', 'attributes.GoodForMeal', 'attributes.DriveThru', 'attributes.HappyHour',
                    'attributes.BusinessAcceptsBitcoin', 'attributes.RestaurantsTableService', 'attributes.Music', 
                    'attributes.BestNights', 'attributes.Smoking', 'attributes.GoodForDancing', 'attributes.Corkage', 
                    'attributes.BYOB', 'attributes.AgesAllowed', 'attributes.BYOBCorkage', 'attributes.DietaryRestrictions',
                    'attributes.AcceptsInsurance', 'attributes.Open24Hours', 'attributes.RestaurantsCounterService', 
                    'attributes.HairSpecializesIn', 'hours', 'attributes', 'categories', 'is_open', 'address']

    df.drop(drop_columns, axis=1, inplace=True)

    logging.info('Fixing \'Noise Level\' ')
    # NoiseLevel
    noise_levels = ['quiet', 'average', 'loud', 'very_loud']

    # Apply the cleaning function to the 'column_name' column
    # bug
    df.loc[:, 'attributes.NoiseLevel'] = df['attributes.NoiseLevel'].apply(clean_category)

    # Filling in missing values with random noise levels
    null_mask = df['attributes.NoiseLevel'].isnull()
    df.loc[null_mask, 'attributes.NoiseLevel'] = df['attributes.NoiseLevel'].apply(lambda x: np.random.choice(noise_levels))

    # Alcohol
    logging.info('Fixing \'Alcohol\' ')

    df.loc[:, 'attributes.Alcohol'] = df['attributes.Alcohol'].apply(clean_category)
    # applying "clean_category" introduces some null values
    df.loc[:, 'attributes.Alcohol'] = df['attributes.Alcohol'].fillna('no')

    # RestaurantPriceRange
    logging.info('Fixing \'RestaurantPriceRange\' ')

    df.loc[:, 'attributes.RestaurantsPriceRange2'] = df['attributes.RestaurantsPriceRange2'].apply(clean_category)

    # filling in missing values with random ratings.
    min_value = 1
    max_value = 4

    mask = df['attributes.RestaurantsPriceRange2'].isnull()
    df.loc[mask, 'attributes.RestaurantsPriceRange2'] = df['attributes.RestaurantsPriceRange2'].apply(lambda x: np.random.randint(min_value, max_value+1))

    # Fix the dtype to 'int'
    df.loc[:, 'attributes.RestaurantsPriceRange2'] = df['attributes.RestaurantsPriceRange2'].astype('int')

    # WiFi
    logging.info('Fixing \'Wifi\' ')
    
    df.loc[:, 'attributes.WiFi'] = df['attributes.WiFi'].apply(clean_category)

    df.loc[:, 'attributes.WiFi'] = df['attributes.WiFi'].fillna('no')

    # Ambience
    logging.info('Fixing \'Ambience\' ')

    # bug
    df.loc[:, 'attributes.Ambience'] = df['attributes.Ambience'].apply(clean_category)

    # Define a default state for this column
    default_ambience_string = "{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': False}"
    # Imputing missing values
    # df['attributes.Ambience'].fillna(default_ambience_string, inplace=True)
    df.loc[:, 'attributes.Ambience'] = df['attributes.Ambience'].fillna(default_ambience_string)

    # Fixing the values in the column
    # bug
    df.loc[:, 'attributes.Ambience'] = df['attributes.Ambience'].apply(lambda x: parse_json_string(x, default='absent'))

    # BikeParking
    assert not df['attributes.BikeParking'].isnull().any(), "AssertionError: Null values found in the BikeParking column."

    # BusinessAcceptsCreditCards
    assert not df['attributes.BusinessAcceptsCreditCards'].isnull().any(), "AssertionError: Null values found in the BusinessAcceptsCreditCards column."
    
    # OutdoorSeating
    assert not df['attributes.OutdoorSeating'].isnull().any(), "AssertionError: Null values found in the OutdoorSeating column."

    # BusinessParking
    logging.info('Fixing \'BusinessParking\' ')

    # bug
    df.loc[:, 'attributes.BusinessParking'] = df['attributes.BusinessParking'].apply(clean_category)

    # No Parking dict map
    noparking_string = "{'garage': False, 'street': False, 'validated': False, 'lot': False, 'valet': False}"

    # Filling the missing values with 'noparking'
    df.loc[:, 'attributes.BusinessParking'] = df['attributes.BusinessParking'].fillna(noparking_string)

    # bug
    df.loc[:, 'attributes.BusinessParking'] = df['attributes.BusinessParking'].apply(lambda x: parse_json_string(x, default='noparking'))

    # RestaurantsGoodForGroups
    assert not df['attributes.RestaurantsGoodForGroups'].isnull().any(), "AssertionError: Null values found in the RestaurantsGoodForGroups column."

    # HasTV
    assert not df['attributes.HasTV'].isnull().any(), "AssertionError: Null values found in the HasTV column."
    
    # RestaurantsReservations
    assert not df['attributes.RestaurantsReservations'].isnull().any(), "AssertionError: Null values found in the RestaurantsReservations column."
    
    assert not df.isnull().any().any(), "AssertionError: Null values found in the final cleaned dataframe."

    return df
  

if __name__ == '__main__':
    args = parse_args()
    input_file = '/raw_data/yelp_academic_dataset_business.json'
    output_file = '/cleaned_data/business.csv'
    # input_file = r'D:\My-Projects\CAPSTONE\dataset\yelp_academic_dataset_business.json'
    # output_file = r'D:\My-Projects\CAPSTONE\delete\business.csv'
    
    cleaned_df = clean_business_dataset(input_file=input_file)

    # Writing dataframe to file
    logging.info(f'Writing Clean file: {output_file}')

    assert not cleaned_df.isnull().any().any(), "AssertionError: Null values found in the final cleaned dataframe."

    cleaned_df.to_csv(output_file, index=False, encoding='utf-8')

    ingest_into_db(cleaned_df, args)

    assert successfully_ingested(cleaned_df, args), "AssertionError: Ingesion into DB, did not write all records.."
    
    logging.info('Successfully exited.')