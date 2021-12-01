# Fynesse Template

This repo is based on the python template repo for doing data analysis according to the Fynesse framework. (https://github.com/lawrennd/fynesse_template)

This repo has been adapted to predict house sale prices by Accessing historical data, Assessing the usefulness of this data, and Addressing the question of house price prediction.

## Access

Gaining access to the data, including overcoming availability challenges (data is distributed across architectures, called from an obscure API, written in log books) as well as legal rights (for example intellectual property rights) and individual privacy rights (such as those provided by the GDPR).

In this section, there is:

 - Code to access and interact with SQL data
   - General code to perform SELECT queries, create a MySQL Connection object etc.
     - Allows for reuse and experimentation later.
   - More specific code to download and insert into SQL database historical House Sale data (https://www.gov.uk/guidance/about-the-price-paid-data) and Postcode data (https://www.getthedata.com/open-postcode-geo)
     - Downloading these on the fly allows for better reuse - automatic download for quick experimentation
   - Also has code to join together these tables to find missing Latitude and Longitude data from Postcodes to House Sale data
     - Important to narrow down data for region first to save time for large amount of sale data
     - Multiple join options to easily visualise data and extend further in the future (by Postcode, within Bounding Box, by District)
  - Code to retrieve Open Street Maps data
    - Get features within a Bounding Box area
      - Gets a variety of Points Of Interest to facilitate many experiments around region.
  - Interaction between SQL and Data Frame formats
    - Again made reusable by taking table name as input.

## Assess

Understanding what is in the data. Is it what it's purported to be, how are missing values encoded, what are the outliers, what does each variable represent and how is it encoded.

In this section, there is:

 - Code to Interact with Street Map features
   - Plot features onto a Map to visualise.
     - Bridges the gap between numbers and names of Points Of Interest to picture. Portrays magnitude and clustering of Points Of Interest within a region.
   - Filter Point Of Interests from Data Frames
     - Facilitate experiments to focus on specific Points Of Interest
 - Visualise House Prices (Note it could be argued this is more Address, but there is inherent inductive bias with the Address question in the background during Assessing. Allows to focus on more relevant Assess questions to best influence final Address question.)
   - Plot histograms, by types for further visualisation of data.
     - Allows to best identify trends in data to best fit models to.
 - Interact with Data Frame (gathered from Access SQL code)
   - Add features, Filter columns.
     - Used to facilitate analysis in Assess stage.
   
## Address

The final aspect of the process is to *address* the question of Predicting House Price data.

In this section, there is:

 - Build design matrices (for training) and feature vectors (for prediction)
   - Open Street Map Points of Interest features. Defaults chosen intuitively to those which would effect house prices (e.g. a Railway nearby causes noise, intuitively reducing the price). Code is extendable for different Points of Interest for future experiments.
     - Originally sampled from Normal distribution of historical data of Points Of Interest for Predictions. Changed to get the Points around the property instead. Although this adds complexity and time to predictions, this makes the features more realistic. I kept this code to insert into Notebook to maintain narrative - future developers may try this method, this will show them why I believe not to.
   - House sale data (Latitude, Longitude, Date, Type). Model can use historical data to (begin to) learn the landscape and hopefully extrapolate this to the new sale prediction. Of course this is unreliable - past performance may differ, landscape may be too complex for linear model, extra features could move model away from better features etc.
   - Nearest sale. Intuitively influences model towards prices in region. Again, past performance may not be representative.
 - Train linear models
   - Generate data using the rest of the zoo of code.
   - Create a model and train on the data
     - Code can be easily reused by changing model type. It could be beneficial in the future to create a function with model type as input to allow future developers to easily experiment.
 - Overall, there is a flaw in the model. There is no description of the house itself apart from the type. E.g. a detached house could be a luxury 10000 sq ft mansion, or a tiny shed. In the future it could be beneficial to join with a house dataset (e.g. planning permission layouts, rental websites) to extract features about the house itself.
