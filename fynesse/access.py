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
import pandas as pd
import osmnx as ox
import datetime

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


def select_top(conn, table, n):
    """
    Query n first rows of the table
    :param conn: the Connection object
    :param table: The table to query
    :param n: Number of rows to query
    :return: Rows of data
    """
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {table} LIMIT {n}')

    rows = cur.fetchall()
    return rows


def select_all(conn, table):
    """
    Query rows of the table
    :param conn: the Connection object
    :param table: The table to query
    :return: Rows of data
    """
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {table}')

    rows = cur.fetchall()
    return rows


def head(conn, table, n=5):
    """
    Query head of the table
    :param conn: the Connection object
    :param table: The table to query
    :param n: Number of rows to query
    """
    rows = select_top(conn, table, n)
    for r in rows:
        print(r)


def upload_data_from_file(conn, filename, table_name, enclosing_char=''):
    """
    Upload file to the SQL table
    :param conn: the Connection object
    :param filename: The name of the file to upload (i.e. the csv)
    :param table_name: Name of table to upload to
    :param enclosing_char: Character enclosing data in csv (e.g.", ', )
    """
    cur = conn.cursor()
    cur.execute(f"""LOAD DATA LOCAL INFILE '{filename}' INTO TABLE `{table_name}`
                    FIELDS TERMINATED BY ',' ENCLOSED BY '{enclosing_char}'
                    LINES STARTING BY '' TERMINATED BY '\n';""")
    conn.commit()


def download_file_from_url(url, file_name=None):
    """
    Download the data from a URL and save it locally
    Named as the file_name passed in, or from the url by default
    :param url: the url to download from
    :param file_name: (OPTIONAL) The name of the file to download (i.e. the csv)
    :return: Filename
    """
    with urllib.request.urlopen(url) as response:
        if file_name is None:
            file_name = url.split("/")[-1]
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(response, f)
    return file_name


def get_pp_url_for_year_part(year, part):
    """
    Return the URL for the relevant Price Paid Data for the relevant year and part
    :param year: Year of data
    :param part: Part of the data
    :return: pp URL for year and part
    """
    data_url_root = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"
    return data_url_root + f"pp-{year}-part{part}.csv"


def get_postcode_url():
    """
    Return the URL for the Postcode Data
    :return: The URL
    """
    return "https://www.getthedata.com/downloads/open_postcode_geo.csv.zip"


def upload_pp_data(conn):
    """
    Upload all of the Price Paid Data to the SQL database
    Splitting it into years and parts
    :param conn: the Connection object
    """
    end_year = datetime.datetime.now().year + 1  # make it usable in the future
    for year in range(1995, end_year):
        # Download and upload each part of the data
        for part in [1, 2]:
            print(f"Downloading data for {year} part {part}...")
            filename = download_file_from_url(get_pp_url_for_year_part(year, part))
            print(f"Download successful. Now uploading to sql database...")
            upload_data_from_file(conn, filename, 'pp_data', enclosing_char="\"")
            print(f"Data successfully uploaded for {year} part {part}")
            print()


def upload_postcode_data(conn):
    """
    Upload the Postcode Data to the SQL database
    :param conn: the Connection object
    """
    print(f"Downloading data Postcode data zip file...")
    filename = download_file_from_url(get_postcode_url())
    print("Download successful. Now unzipping...")
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(".")
    print("Unzipped successfully. Now uploading to sql database...")
    upload_data_from_file(conn, "open_postcode_geo.csv", 'postcode_data')
    print("Data successfully uploaded.")
    print()


def truncate_table(conn, table, commit=False):
    cur = conn.cursor()
    print("Truncating table (clear data while preserving structure)...")
    cur.execute(f"TRUNCATE TABLE {table}")  # clear the data, preserve structure
    print("Table truncated")

    if commit:
        conn.commit()


def join_data_for_postcode_time(conn, postcode_start, date_of_transfer, truncate=True):
    """
    Select the data for the region with postcode starting with postcode_start,
    join into one table, and store the results in prices_coordinates_data
    :param conn: the Connection object
    :param postcode_start: the Postcode Prefix
    :param date_of_transfer: the Prefix for the date of transfer
    """
    cur = conn.cursor()
    if truncate:
        truncate_table(conn, "prices_coordinates_data", commit=True)
    print("Now inserting data for", postcode_start, date_of_transfer)
    cur.execute(f"""
        INSERT INTO prices_coordinates_data
            (price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality,
            town_city, district, county, country, lattitude, longitude, db_id)
        SELECT price, date_of_transfer, pp.postcode, property_type, new_build_flag, tenure_type, locality,
            town_city, district, county, country, lattitude, longitude, pp.db_id
        FROM
            (SELECT * FROM pp_data WHERE pp_data.postcode LIKE '{postcode_start}%'
                AND pp_data.date_of_transfer LIKE '{date_of_transfer}%') pp
        INNER JOIN
            (SELECT * FROM postcode_data WHERE postcode_data.postcode LIKE '{postcode_start}%') pc
        ON pp.postcode = pc.postcode \n;""")
    conn.commit()


def join_date_for_postcodes_time(conn, postcodes, date):
    for postcode in postcodes:
        print("Inserting data for postcode", postcode)
        join_data_for_postcode_time(conn, postcode, date, truncate=False)


def join_data_for_postcodes_times(conn, postcodes, dates):
    truncate_table(conn, "prices_coordinates_data", commit=True)
    for date in dates:
        print("Inserting data for date", date)
        join_date_for_postcodes_time(conn, postcodes, date)

    conn.commit()


def join_data_within_bbox_time(conn, north, south, east, west, date_of_transfer, truncate=True):
    cur = conn.cursor()
    if truncate:
        truncate_table(conn, "prices_coordinates_data", commit=True)
    print("Now inserting data from bounding box", [north, south, east, west], "for year", date_of_transfer)
    cur.execute(f"""
            INSERT INTO prices_coordinates_data
                (price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality,
                town_city, district, county, country, lattitude, longitude, db_id)
            SELECT price, date_of_transfer, pp.postcode, property_type, new_build_flag, tenure_type, locality,
                town_city, district, county, country, lattitude, longitude, pp.db_id
            FROM
                (SELECT * FROM pp_data WHERE pp_data.date_of_transfer LIKE '{date_of_transfer}%') pp
            INNER JOIN
                (SELECT * FROM postcode_data WHERE {west} <= postcode_data.longitude AND postcode_data.longitude < {east}
                        AND {south} <= postcode_data.lattitude AND postcode_data.lattitude < {north}) pc
            ON pp.postcode = pc.postcode \n;""")
    conn.commit()


def join_data_within_bbox_times(conn, bbox, dates):
    truncate_table(conn, "prices_coordinates_data", commit=True)
    for date in dates:
        join_data_within_bbox_time(conn, *bbox, date, truncate=False)
    conn.commit()


def join_data_for_region_time(conn, region, date_of_transfer, truncate=True):
    """
    Select the data for the region with postcode starting with postcode_start,
    join into one table, and store the results in prices_coordinates_data
    :param conn: the Connection object
    :param region: the District Prefix
    :param date_of_transfer: the Prefix for the date of transfer
    """
    cur = conn.cursor()
    if truncate:
        truncate_table(conn, "prices_coordinates_data", commit=True)
    cur.execute(f"""
        INSERT INTO prices_coordinates_data
            (price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality,
            town_city, district, county, country, lattitude, longitude, db_id)
        SELECT price, date_of_transfer, pp.postcode, property_type, new_build_flag, tenure_type, locality,
            town_city, district, county, country, lattitude, longitude, pp.db_id
        FROM
            (SELECT * FROM pp_data WHERE pp_data.district = '{region}%'
                AND pp_data.date_of_transfer LIKE '{date_of_transfer}%') pp
        INNER JOIN
            postcode_data pc
        ON pp.postcode = pc.postcode \n;""")
    conn.commit()


def clean_place_name(place_name):
    """
    Clean the place name, replacing spaces with dashes and removing commas
    :param place_name: Place name to clean
    :return: cleaned place name
    """
    return place_name.lower().replace(' ', '-').replace(',', '')


def sql_to_df(conn, table_name):
    """
    Convert SQL data to a pandas df
    :param conn: Connection object
    :param table_name: name of table to get data from
    :return: the data from that table as a pandas df
    """
    return pd.read_sql(f'SELECT * FROM {table_name}', conn)


def get_points_within_bb(df, bb):
    """
    Get the points in a dataframe within a bounding box
    :param df: dataframe
    :param bb: bounding box
    :return: points with lat and long in bounding box
    """
    north, south, east, west = bb
    return df.loc[(west <= df["longitude"]) & (df["longitude"] < east) & (south <= df["lattitude"]) &
                  (df["lattitude"] < north)]


def get_within_bb_for_region(conn, bb):
    """
    Get points within a bounding box for the data held in the prices_coordinates_data table
    :param conn: Connection object
    :param bb: bounding box
    :return: points withini lat and long in bounding box
    """
    return get_points_within_bb(sql_to_df(conn, "prices_coordinates_data"), bb)


def get_lat_from_postcode_year(conn, postcode, year):
    """
    Get latitude and longitude from postcode and year
    By Merging the relevant data into one table
    :param conn: Connection object
    :param postcode: postcode prefix
    :param year: year of purchase
    :return: mean of latitude and longitude for this postcode region
    """
    join_data_for_postcode_time(conn, postcode, year)
    df = sql_to_df(conn, "prices_coordinates_data")
    lat = df['lattitude']
    lon = df['longitude']
    return lat.mean(), lon.mean(), lat.min(), lat.max(), lon.min(), lon.max()


def get_bounding_box(lat, lon, width=0.02, height=0.02):
    """
    Get bounding box around a latitude and longitude
    :param lat: latitude
    :param lon: longitude
    :param width: width of bounding box
    :param height: height of bounding box
    :return: North, South, East, West of bounding box
    """
    return lat + height / 2, lat - height / 2, lon + width / 2, lon - width / 2


def get_lat_lon_for(place_name):
    """
    Get the lat lon pair for the geocode place name
    :param place_name: place name
    :return: latitude and longitude for this place
    """
    return ox.geocoder.geocode(place_name)


def get_points_of_interest(north, south, east, west, tags):
    """
    Get points of interest in a bounding box
    :param north: North point of bounding box
    :param south: South point of bounding box
    :param east: East point of bounding box
    :param west: West point of bounding box
    :param tags: tags to lookup place name, see defaults in assess.py
    :return: Points of interest in this region according to the tags
    """
    return ox.geometries_from_bbox(north, south, east, west, tags)


def get_gdf_data_for(place_name, bb_width=0.02, bb_height=0.02):
    """
    Get relevant geodataframe for the place name
    :param place_name: place name
    :param bb_width: bounding box width
    :param bb_height: bounding box height
    :return: nodes, edges, area of graph for region, and bounding box
    """
    # Get lat, lon for place_name
    place_name = clean_place_name(place_name)
    lat, lon = get_lat_lon_for(place_name)

    # Create bounding box for area
    north, south, east, west = get_bounding_box(lat, lon, width=bb_width, height=bb_height)

    # Get graph data
    graph = ox.graph_from_bbox(north, south, east, west)
    nodes, edges = ox.graph_to_gdfs(graph)
    area = ox.geocode_to_gdf(place_name)
    return nodes, edges, area, [north, south, east, west]


def select_postcodes_within_bbox(conn, north, south, east, west, limit=6):
    print("Getting postcodes within the bounding box", [north, south, east, west])
    cur = conn.cursor()
    cur.execute(f"""SELECT DISTINCT postcode_district FROM postcode_data
                    WHERE {west} <= longitude and longitude < {east}
                        and {south} <= lattitude and lattitude < {north}""")

    rows = cur.fetchall()

    # clean up into list of postcodes - returns data in form (('CB1',),('CB2',),...)
    return [postcode_tuple[0] for postcode_tuple in rows][:limit]  # could remove furthest postcodes instead


def get_date_range(date, n=1):
    # Get date range, +- n years if still in range
    date = int(date)
    date_range = [date]
    date_range_size = n
    for date_below in [date - (i + 1) for i in range(date_range_size)]:
        if date_below >= 1995:
          date_range.append(date_below)
    for date_above in [date + (i + 1) for i in range(date_range_size)]:
        if date_above <= 2021:
          date_range.append(date_above)
        date_range.sort()
    return date_range
