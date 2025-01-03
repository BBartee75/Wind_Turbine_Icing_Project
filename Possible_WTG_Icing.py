###################################################################################################
#
#' Name: Wind Farm Icing Possible Data Analysis
#
#' Author: Bently Bartee
#' Date: 10/12/2024
#
#' Version: 1.0
#' Description: This script fetches data from a SQL database and processes it to identify possible 
#               icing conditions for wind turbines.
#
###################################################################################################



# Python libraries needed for the script
# --------------------------------------------------------------------------------------------------
# Check if required packages are installed and install them if missing
import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ['pyodbc', 'pandas', 'numpy', 'sqlalchemy']
# Install missing packages
for package in required_packages:
    install_if_missing(package)

# Now import the modules
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
from sqlalchemy import create_engine


## FUNCTIONS NEEDED FOR SCRIPT ##
# -------------------------------------------------------------------------------------------------
# Function to download data from databse for a specific WTG and certain date range
def fetch_data(wtg, start, end):
    """
    Fetch data from the SQL database for a specific WTG.
    """
    server = 'XXXXXX'
    database = 'XXXXDB2'
    connection_string = f'mssql+pyodbc://{server}/{
        database}?driver=ODBC+Driver+17+for+SQL+Server'

    query_template = f"""
    SELECT
    YEAR(A.PCTimeStamp) as Year,
    MONTH(A.PCTimeStamp) as Month,
    DAY(A.PCTimeStamp) as Day,
    A.PCTimeStamp,
    '{wtg}' as WTG, -- Placeholder for WTG value
    A.Amb_Temp_Avg,
    A.Amb_WindSpeed_Max,
    A.Amb_WindSpeed_Avg,
    A.Amb_WindDir_Abs_Avg,
    A.Nac_Direction_Avg,
    A.Grd_Prod_Pwr_Avg,
    A.Sys_Stats_TrbStat,
    A.Sys_Logs_FirstActAlarmNo,
    A.Blds_PitchAngle_Avg
    FROM [XXXXDB2].[dbo].[T_{wtg}_AP10MinData] as A
    WHERE
        A.Amb_WindSpeed_Avg > 8
        AND A.PCTimeStamp >= '{start}'
        AND A.PCTimeStamp <= '{end}';
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        # Use the engine to fetch data
        SPData = pd.read_sql_query(query_template, engine)
        return SPData
    except Exception as e:
        print(f"Error querying table T_{wtg}_AP10MinData: {e}")
        return pd.DataFrame()

# Use parallel processing to fetch data for all WTGs and date ranges
def fetch_all_data(Q_tables, tables):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, row in Q_tables.iterrows():
            start = row['Start_Date']
            end = row['End_Date']
            for table_name in tables['TABLE_NAME']:
                wt = table_name.split('_')[1]
                futures.append(executor.submit(fetch_data, wt, start, end))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return pd.concat(results, ignore_index=True)

# Function to filter a DataFrame to retain only 10-minute intervals in chronological order
def filter_10min_intervals(df):
    """
    Filters a DataFrame to retain only 10-minute intervals in chronological order.    
    """
    # Check if required columns exist
    required_columns = ['WTG', 'PCTimeStamp']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            "DataFrame must contain 'WTG' and 'PCTimeStamp' columns")
    # Convert PCTimeStamp to datetime
    df['PCTimeStamp'] = pd.to_datetime(df['PCTimeStamp'])
    # Sort by WTG and PCTimeStamp
    df = df.sort_values(['WTG', 'PCTimeStamp'])
    # Calculate time difference in minutes
    df['TimeDiff'] = df.groupby(
        'WTG')['PCTimeStamp'].diff().dt.total_seconds() / 60
    # Check if time difference is approximately 10 minutes
    df['Is10MinInterval'] = np.isclose(df['TimeDiff'], 10, atol=1)
    # Shift Is10MinInterval column by 1
    df['Is10MinIntervalPrevious'] = df.groupby(
        'WTG')['Is10MinInterval'].shift()
    # Assign 1 if previous row is in datetime order, else 2
    df['InOrder'] = np.where(df['Is10MinIntervalPrevious'], 1, 2)
    # Create a new column to track the start of each 10-minute interval
    df['StartOfInterval'] = np.where(
        (df['InOrder'] == 1) & (df['Is10MinInterval']), True, False)
    # Find the index of the last row in each group
    last_row_index = df.groupby('WTG')['StartOfInterval'].cumsum().idxmax()
    # Calculate the time difference between consecutive 10-minute intervals
    df['TimeDiffBetweenIntervals'] = df.groupby(
        'WTG')['PCTimeStamp'].diff().dt.total_seconds() / 60
    # Filter for rows where the next interval starts immediately after
    df['ValidInterval'] = np.where((df['InOrder'] == 1) &
                                   (df['Is10MinInterval'] == True) &
                                   (df.index.get_level_values(0) < last_row_index),
                                   1, 0)
    # Drop intermediate columns
    columns_to_drop = ['TimeDiff', 'Is10MinInterval',
                       'Is10MinIntervalPrevious', 'InOrder', 'StartOfInterval']
    df = df.drop(columns_to_drop, axis=1)
    # Filter out intervals that are not in chronological order
    df = df[df['ValidInterval'] == 1].reset_index(drop=True)
    return df

#  Process wind turbine data to find Icing possbile in running and non-running modes.
def process_Icedata(wtg, start, end):
    """
    Process wind turbine data and calculate various metrics to find Icing possbile.    
    """
    print(f"Fetching data for WTG: {wtg}")

    # Fetch data for the specific WTG
    result_df = fetch_data(wtg, start, end) #enter you start and end date here
    # Check if input DataFrame is empty or contains NaN values
    if result_df.empty or result_df.isnull().any().any():
        raise ValueError(
            "Input DataFrame must not be empty and contain no NaN values")

    result_df2 = result_df
    # Filters a DataFrame to retain only 10-minute intervals in chronological order.
    if wtg != 207317:
        result_df2 = filter_10min_intervals(result_df2)
    print(f"Processing Ice data for: {wtg}")

    # add Event_Type Column----------------------------------------------
    # filter date for customer outages
    Miss_file = r'C:\Users\bbartee\Desktop\SP\South Plains Outages - GE ROC Log.xlsx'
    Miss_dates = pd.read_excel(Miss_file)
    Miss_dates['Start Date & Time:'] = pd.to_datetime(
        Miss_dates['Start Date & Time:'])
    Miss_dates['End Date & Time:'] = pd.to_datetime(
        Miss_dates['End Date & Time:'])
    Miss_dates = Miss_dates.iloc[:, [2, 5, 6]]
    # Sort both dataframes by their timestamp columns
    result_df2 = result_df2.sort_values('PCTimeStamp')
    Miss_dates = Miss_dates.sort_values('Start Date & Time:')

    # Create a function to check if timestamp falls within range
    def check_time_range(row):
        mask = (Miss_dates['Start Date & Time:'] <= row['PCTimeStamp']) & \
            (Miss_dates['End Date & Time:'] >= row['PCTimeStamp'])
        if any(mask):
            return Miss_dates.loc[mask.idxmax(), 'Event Type:']
        return None
    
    # Add events column
    result_df2['Event_Type'] = result_df2.apply(check_time_range, axis=1)
    result_df2['Event_Type'] = result_df2['Event_Type'].fillna('none')

    # Find possible icing Condition - Running
    # ------------------------------------------------------------------    
    # see: https://www.sciencedirect.com/science/article/abs/pii/S096014810900408X
    # see: https://theconversation.com/the-science-behind-frozen-wind-turbines-and-how-to-keep-them-spinning-through-the-winter-156520
    # look for a conditon where temp is <= 3 C, and power out up is 15% less than OEM power curve output per wind bin
    # temps freezing or below (between 3 C & -17 C)
    # blade angle less then 25 degree
    def round_to_half(x):
        return round(x * 2) / 2
    icing_df = result_df2.iloc[:, [3, 4, 5, 7, 10, 11, 12, 13, -1]]
    icing_df = icing_df[(icing_df['Amb_Temp_Avg'] <= 3) &  # temps freezing or below (<= 3C)
                        (icing_df['Amb_Temp_Avg'] >= -17) &
                        (icing_df['Blds_PitchAngle_Avg'] <= 25)
                        ]
    icing_df = filter_10min_intervals(icing_df)
    PC_file = r'C:\Users\bbartee\Desktop\SP\UL_SouthPlaines_Edgewise\V100_PC.xlsx'
    V100PC = pd.read_excel(PC_file)
    # add wind speed wind bins
    icing_df['WindBin'] = icing_df['Amb_WindSpeed_Avg'].apply(round_to_half)
    icing_df = pd.merge(
        icing_df, V100PC[['WindBin', 'Power']], on='WindBin', how='left')
    icing_df['Poss_Ice_Run'] = 'N/A'
    # look for unequal power; where power output is < 15% of rated power from OEM power curve by Wind Bin.
    icing_df['Poss_Ice_Run'] = np.where(icing_df['Grd_Prod_Pwr_Avg'] < (
        icing_df['Power'] - (icing_df['Power'] * 0.15)), 'icing_poss', icing_df['Poss_Ice_Run'])
    # merge icing data to result_df
    icing_Run = pd.merge(result_df2, icing_df[['WTG', 'PCTimeStamp', 'Poss_Ice_Run']], on=[
                         'WTG', 'PCTimeStamp'], how='left')
    icing_Run['Poss_Ice_Run'] = icing_Run['Poss_Ice_Run'].fillna('no_ice')
    icing_Run = icing_Run[(icing_Run['Poss_Ice_Run'] == 'icing_poss')]

    # Find possible icing Condition - NOT Running
    # ------------------------------------------------------------------    
    # see: https://www.sciencedirect.com/science/article/abs/pii/S096014810900408X
    # see: https://theconversation.com/the-science-behind-frozen-wind-turbines-and-how-to-keep-them-spinning-through-the-winter-156520
    # look for a conditon where temp is <= 3 C, and power out up is 15% less than OEM power curve output per wind bin
    # temps freezing or below (between 3 C & -17 C)
    # Grd_Prod_Pwr_Avg = 0 kw
    # blade angle greater then 25 degree
    def round_to_half(x):
        return round(x * 2) / 2
    icing_df = result_df2.iloc[:, [3, 4, 5, 7, 10, 11, 12, 13, -1]]
    icing_df = icing_df[(icing_df['Amb_Temp_Avg'] <= 3) &  # temps freezing or below (<= 3C)
                        (icing_df['Amb_Temp_Avg'] >= -17) &
                        (icing_df['Blds_PitchAngle_Avg'] >= 25)
                        ]
    icing_df = filter_10min_intervals(icing_df)
    PC_file = r'C:\Users\bbartee\Desktop\SP\UL_SouthPlaines_Edgewise\V100_PC.xlsx'
    V100PC = pd.read_excel(PC_file)
    # add wind speed wind bins
    icing_df['WindBin'] = icing_df['Amb_WindSpeed_Avg'].apply(round_to_half)
    icing_df = pd.merge(
        icing_df, V100PC[['WindBin', 'Power']], on='WindBin', how='left')
    icing_df['Poss_Ice_NoRun'] = 'N/A'
    # look for unequal power; where power output is < 15% of rated power from OEM power curve by Wind Bin.
    icing_df['Poss_Ice_NoRun'] = np.where(icing_df['Grd_Prod_Pwr_Avg'] < (
        icing_df['Power'] - (icing_df['Power'] * 0.15)), 'icing_poss', icing_df['Poss_Ice_NoRun'])
    # merge icing data to result_df
    icing_NotRun = pd.merge(result_df2, icing_df[[
                            'WTG', 'PCTimeStamp', 'Poss_Ice_NoRun']], on=['WTG', 'PCTimeStamp'], how='left')
    icing_NotRun['Poss_Ice_NoRun'] = icing_NotRun['Poss_Ice_NoRun'].fillna(
        'no_ice')
    icing_NotRun = icing_NotRun[(icing_NotRun['Poss_Ice_NoRun'] == 'icing_poss') &
                                (icing_NotRun['Grd_Prod_Pwr_Avg'] < 0)
                                ]    
    # Merge with icing_NotRun to add Poss_Ice_NoRun
    Icing_All = pd.merge(result_df2, icing_NotRun[[
                         'WTG', 'PCTimeStamp', 'Poss_Ice_NoRun']], on=['WTG', 'PCTimeStamp'], how='left')
    # Merge with icing_Run to add Poss_Ice_Run
    Icing_All = pd.merge(Icing_All, icing_Run[['WTG', 'PCTimeStamp', 'Poss_Ice_Run']], on=[
                         'WTG', 'PCTimeStamp'], how='left')
    Icing_All['Poss_Ice_NoRun'] = Icing_All['Poss_Ice_NoRun'].fillna('no_ice')
    Icing_All['Poss_Ice_Run'] = Icing_All['Poss_Ice_Run'].fillna('no_ice')
    Icing_result = Icing_All

    return Icing_result


# --------------------------------------------------------------------------------------------------
# Fetch WTG data and process data for icing conditions
# --------------------------------------------------------------------------------------------------
WTGs = (207341, 207294, 207346, 207297, 207295, 207339,
        207363, 207292, 207301, 207293, 207371, 207353,
        207348, 207338, 207359, 207344, 207350, 207356,
        207358, 207298, 207289, 207330, 207333, 207367,
        207328, 207364, 207291, 207316, 207342, 207357,
        207361, 207299, 207319, 207340, 207335, 207369,
        207347, 207368, 207351, 207373, 207315, 207334,
        207360, 207324, 207355, 207345, 207365, 207317,
        207349, 207318, 207321, 207332, 207300, 207308,
        207312, 207326, 207310, 207307, 207372, 207362,
        207374, 207337, 207306, 207313, 207286, 207327,
        207311, 207320, 207343, 207322, 207284, 207323,
        207296, 207282, 207290, 207283, 207276, 207366,
        207280, 207302, 207329, 207288, 207285, 207277,
        207305, 207281, 207303, 207336, 207304, 207325,
        207278, 207309, 207354, 207314, 207287, 207375,
        207352, 207279, 207370, 207331)

# Create a new DataFrame for counting 'ice_poss' and 'no_ice' values
Icing_result = pd.DataFrame()
start = '2018-01-01 00:00:00' #enter you start date here
end = '2024-01-02 00:00:00'   #enter you end date here

for iteration_number, wtg in enumerate(WTGs, start=1):
    print(f'Iteration Number: {iteration_number}, WTG: {wtg}')
    result = process_Icedata(wtg, start, end)
    Icing_result = pd.concat([Icing_result, result], ignore_index=True)

# Add columns for Icing_NoRun and Icing_Run
Icing_result['Icing_NoRun'] = Icing_result['Poss_Ice_NoRun'].apply(
    lambda x: 1 if x == 'icing_poss' else 0)
Icing_result['Icing_Run'] = Icing_result['Poss_Ice_Run'].apply(
    lambda x: 1 if x == 'icing_poss' else 0)


icing = Icing_result.head()

# Initialize Event_Type column with 'none'
Icing_result['Event_Type'] = 'none'
# Process in chunks to avoid memory error
chunk_size = 100000  # Adjust this value based on your available memory

print("Processing chunks...")
for start_idx in range(0, len(Icing_result), chunk_size):
    # Print progress
    print(f"Processing rows {start_idx} to {
          min(start_idx + chunk_size, len(Icing_result))}")
    # Process chunk
    end_idx = min(start_idx + chunk_size, len(Icing_result))
    chunk = Icing_result.iloc[start_idx:end_idx]
    # Apply function to chunk and update original DataFrame
    Icing_result.iloc[start_idx:end_idx, Icing_result.columns.get_loc('Event_Type')] = \
        chunk.apply(check_time_range, axis=1)
print("Processing complete!")


# Save the final DataFrame to a CSV file
Icing_result.to_csv(
    'C:/Users/XXXX/Desktop/SP/Icing_result_Rev7.csv', index=False)


