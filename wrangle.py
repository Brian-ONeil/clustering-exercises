#######IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import os
import env
import wrangle as wra

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#######FUNCTIONS

zillow_query = """
        select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt,
        taxamount, fips
        from properties_2017
        where propertylandusetypeid = '261';
        """

def new_zillow_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a db_url to mySQL
    - return a df of the given query from the telco_db
    """
    url = env.get_db_url('zillow')
    
    return pd.read_sql(SQL_query, url)

def get_zillow_data(SQL_query, filename = 'zillow.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv: defaulted to telco.csv
    - outputs iris df
    """
    
    if os.path.exists(filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_zillow_data(SQL_query)

        df.to_csv(filename)
        return df

### new functions

def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing

def nulls_by_row(df, index_id = 'parcelid'):
    """
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': pct_miss})

    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True).reset_index()[[index_id, 'num_cols_missing', 'percent_cols_missing']]
    
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    # distribution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head(3)}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
    _______________________________________""")                   
        
    fig, axes = plt.subplots(1, len(get_numeric_cols(df)), figsize=(15, 5))
    
    for i, col in enumerate(wra.get_numeric_cols(df)):
        sns.histplot(df[col], ax = axes[i])
        axes[i].set_title(f'Histogram of {col}')
    plt.show()
    
### cleaning data (prepare)

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df

def wrangle_zillow(df):
    '''takes in df, cleans df, uses function to handle_missing_values'''
    
    # Drop 'Unnamed: 0' column
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Define the values to drop
    values_to_drop = ['Planned Unit Development', 'Triplex (3 Units, Any Combination)', 
                      'Quadruplex (4 Units, Any Combination)', 'Cluster Home', 
                      'Commercial/Office/Residential Mixed Used', 'Cooperative']

    # Drop the rows with the specified values in the 'propertylandusedesc' column
    df = df[~df['propertylandusedesc'].isin(values_to_drop)]
    
    # see function above
    df = wra.handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75)
    
    # Drop duplicates in the 'parcelid' column
    df.drop_duplicates(subset='parcelid', inplace=True)
    
    # Replace all 'nan' values in the 'heatingorsystemdesc' column with 'Yes'
    df ['heatingorsystemdesc'] = df['heatingorsystemdesc'].fillna('Yes')
    
    # Replace all 'nan' values in the 'heatingorsystemtypeid' column with '24'
    df['heatingorsystemtypeid'] = df['heatingorsystemtypeid'].fillna(24)
    
    # Drop the specified columns from the dataframe
    df = df.drop(['buildingqualitytypeid', 'propertyzoningdesc', 'unitcnt'], axis=1)
    
    # Replace all NaN values with 0 in the 'lotsizesquarefeet' column where the 'propertylandusedesc' column has 'Condominium'
    condo_idx = df[df['propertylandusedesc'] == 'Condominium'].index
    df.loc[condo_idx, 'lotsizesquarefeet'] = df.loc[condo_idx, 'lotsizesquarefeet'].fillna(0)
    
    df = df.dropna()
   
    return df


def data_prep(df, col_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - list of columns
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - removes unwanted columns
    - remove rows and columns that contain a high proportion of missing values
    - returns cleaned df
    """
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df

def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    train, test = train_test_split(df,
                                   test_size=.2,
                                   random_state=123,
                                   )
    train, validate = train_test_split(train,
                                       test_size=.25,
                                       random_state=123,
                                       )
    return train, validate, test

def get_X_train_val_test(train,validate, test, x_target, y_target):
    '''
    geting the X's and y's and returns them
    '''
    X_train = train.drop(columns = x_target)
    X_validate = validate.drop(columns = x_target)
    X_test = test.drop(columns = x_target)
    y_train = train[y_target]
    y_validate = validate[y_target]
    y_test = test[y_target]
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def scaler_robust(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the RobustScaler on it
    '''
    scaler = RobustScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def scaled_data_to_dataframe(X_train, X_validate, X_test):
    '''
    This function scales the data and returns it as a pandas dataframe
    '''
    X_train_columns = X_train.columns
    X_validate_columns = X_validate.columns
    X_test_columns = X_test.columns
    X_train_numbers, X_validade_numbers, X_test_numbers = scaler_robust(X_train, X_validate, X_test)
    X_train_scaled = pd.DataFrame(columns = X_train_columns)
    for i in range(int(X_train_numbers.shape[0])):
        X_train_scaled.loc[len(X_train_scaled.index)] = X_train_numbers[i]
    X_validate_scaled = pd.DataFrame(columns = X_validate_columns)
    for i in range(int(X_validade_numbers.shape[0])):
        X_validate_scaled.loc[len(X_validate_scaled.index)] = X_validade_numbers[i]
    X_test_scaled = pd.DataFrame(columns = X_test_columns)
    for i in range(int(X_test_numbers.shape[0])):
        X_test_scaled.loc[len(X_test_scaled.index)] = X_test_numbers[i]
    return X_train_scaled, X_validate_scaled, X_test_scaled


def new_mall_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a db_url to mySQL
    - return a df of the given query from the mall_customers db
    """
    url = env.get_db_url('mall_customers')
    
    return pd.read_sql(SQL_query, url)




def get_mall_data(SQL_query, filename = 'mall_customers.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv: defaulted to mall_customers.csv
    - outputs mall_customers df
    """
    
    if os.path.exists(filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_mall_data(SQL_query)

        df.to_csv(filename)
        return df
    








    