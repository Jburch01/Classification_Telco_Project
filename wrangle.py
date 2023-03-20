from env import get_db_url
import pandas as pd
from sklearn.model_selection import train_test_split

# Accuire data from sql server
def get_telco_data():
    '''
    Function will try to return ad database from csv file if file is local and in same directory.
    IF file doesn't exist it will create and store in same directory
    Otherwise will pull from codeup database.
    Must have credentials for codeup database.
    '''
    try:
        csv_info = pd.read_csv('telco_churn.csv', index_col=0 )
        return csv_info
    except FileNotFoundError:
        url = get_db_url('telco_churn')
        info = pd.read_sql('''
            select * from customers
        join contract_types using(contract_type_id)
        join internet_service_types using(internet_service_type_id)
        join payment_types using (payment_type_id);
        ''', url)
        info.to_csv("telco_churn.csv", index=True)
        return info

# Prepping the data for train validate and test
def prep_telco():
    # Get telco db
    telco_db = get_telco_data()
    # change dtype of total charges
    telco_db['total_charges'] = (telco_db['total_charges']) + '0'
    telco_db['total_charges'] = telco_db['total_charges'].astype(float)
    # drop columns
    telco_db.drop(['payment_type_id', 'internet_service_type_id', 'contract_type_id'], axis=1, inplace=True)
    telco_db = pd.concat(
        [telco_db,
         pd.get_dummies((telco_db.select_dtypes(include='object').drop(columns='customer_id')))],
        axis=1)
    return telco_db

# Train Validate and Test split function


def train_validate_test(df, target):
    train_val, test = train_test_split(df,
                                       train_size=0.8,
                                       random_state=706,
                                       stratify=df[target])
    train, validate = train_test_split(train_val,
                                       train_size=0.7,
                                       random_state=706,
                                       stratify=train_val[target])
    return train, validate, test