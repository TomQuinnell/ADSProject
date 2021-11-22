from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""
import urllib.request
import shutil
import pymysql
import zipfile

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


def upload_data_from_file(conn, filename, table_name, enclosing_char=''):
    cur = conn.cursor()
    cur.execute(f"""LOAD DATA LOCAL INFILE '{filename}' INTO TABLE `{table_name}`
                    FIELDS TERMINATED BY ',' ENCLOSED BY '{enclosing_char}'
                    LINES STARTING BY '' TERMINATED BY '\n';""")
    conn.commit()


def download_file_from_url(url, file_name=None):
    """ Download the data from a URL and save it locally
        Named as the file_name passed in, or from the url by default """
    with urllib.request.urlopen(url) as response:
        if file_name is None:
            file_name = url.split("/")[-1]
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(response, f)
    return file_name


def get_pp_url_for_year_part(year, part):
    """ Return the URL for the relevant Price Paid Data
        for the relevant year and part """
    data_url_root = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"
    return data_url_root + f"pp-{year}-part{part}.csv"


def get_postcode_url():
    """ Return the URL for the Postcode Data """
    return "https://www.getthedata.com/downloads/open_postcode_geo.csv.zip"


def upload_pp_data(conn):
    """ Upload the Price Paid Data to the SQL database """
    for year in range(1995, 2022):
        # Download and upload each part of the data
        for part in [1, 2]:
            print(f"Downloading data for {year} part {part}...")
            filename = download_file_from_url(get_pp_url_for_year_part(year, part))
            print(f"Download successful. Now uploading to sql database...")
            upload_data_from_file(conn, filename, 'pp_data', enclosing_char="\"")
            print(f"Data successfully uploaded for {year} part {part}")
            print()


def upload_postcode_data(conn):
    """ Upload the Postcode Data to the SQL database """
    print(f"Downloading data Postcode data zip file...")
    filename = download_file_from_url(get_postcode_url())
    print("Download successful. Now unzipping...")
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(".")
    print("Unzipped successfully. Now uploading to sql database...")
    upload_data_from_file(conn, "open_postcode_geo.csv", 'postcode_data')
    print("Data successfully uploaded.")
    print()


def join_data_for_region_time(conn, region, time):
    """ Select the data for the region, join into one table, and store the results in prices_coordinates_data """
    """select a.item,a.description,a.qty_sold,a.date_sold,a.sold _price,
a.currprice,a.[total sales],a.sale_price,b.price,b.pricedate
into MyNewTable
from ccsales_test a
inner join
ccsales_pricehist2 b
on a.item =b.item;"""
    cur = conn.cursor()
    print("Truncating table...")
    cur.execute("TRUNCATE TABLE prices_coordinates_data")  # clear the data, preserve structure
    print("Table truncated, now inserting data for ", region, time)
    cur.execute(f"""
        INSERT INTO prices_coordinates_data
        (price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality,
        town_city, district, county, country, lattitude, longitude, db_id)
        SELECT price, date_of_transfer, pp.postcode, property_type, new_build_flag, tenure_type, locality,
        town_city, district, county, country, lattitude, longitude, pp.db_id
        FROM (SELECT * FROM pp_data WHERE pp_data.district = "{region}" 
            AND pp_data.date_of_transfer LIKE '{time}%') pp
        INNER JOIN
        postcode_data pc
        ON pp.postcode = pc.postcode \n;""")
    conn.commit()
