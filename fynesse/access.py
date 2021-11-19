from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""
import urllib.request
import shutil
import pymysql

# This file accesses the data

""" Place commands in this file to access the data electronically.
    Don't remove any missing values, or deal with outliers.
    Make sure you have legalities correct, both intellectual property and personal data privacy rights.
    Beyond the legal side also think about the ethical issues around this data. """

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def select_top(conn, table,  n):
    """
    Query n first rows of the table
    :param conn: the Connection object
    :param table: The table to query
    :param n: Number of rows to query
    """
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {table} LIMIT {n}')

    rows = cur.fetchall()
    return rows


def head(conn, table, n=5):
  rows = select_top(conn, table, n)
  for r in rows:
      print(r)


def upload_data_from_file(conn, filename, table_name):
    cur = conn.cursor()
    cur.execute(f"""LOCAL DATA LOAD INFILE '{filename}' INTO TABLE `{table_name}`
                    FIELDS TERMINATED BY ',' 
                    LINES STARTING BY '' TERMINATED BY '\n';""")


def download_file_from_url(url, file_name=None):
    """ Download the data from a URL and save it locally
        Named as the file_name passed in, or from the url by default """
    with urllib.request.urlopen(url) as response:
        if file_name is None:
            file_name = url.split("/")[:1]
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(response, f)


def get_pp_url_for_year_part(year, part):
    """ Return the URL for the relevant Price Paid Data
        for the relevant year and part """
    data_url_root = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"
    return data_url_root + f"pp-{year}-part{part}.csv"


def upload_pp_data(conn):
    """ Upload the Price Paid Data to the SQL database """
    for year in range(1995, 2022):
        # Download and upload each part of the data
        for part in [1, 2]:
            print(f"Downloading data for {year} part {part}...")
            filename = download_file_from_url(get_pp_url_for_year_part(year, part))
            print(f"Download successful. Now uploading to sql database...")
            upload_data_from_file(conn, filename, 'pp_data')
            print(f"Data successfully uploaded for {year} part {part}")
            print()



def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

