# General packages
import pandas as pd
import os
from datetime import datetime
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Unused
# from model1 import closed_form_fit,closed_form_predict
# from sklearn.linear_model import LinearRegression
# from numpy.linalg import inv


def prepData():
    #Feature data and actual wind power production:
    cwd = os.getcwd()
    print(cwd)
    """ Load actual wind power from cwd """
    temp_dir = os.path.join(cwd, 'Actual wind power.csv')
    df = pd.read_csv(temp_dir, sep=";")

    ### Create time index to match feature data
    df["HourUTC"] = df['Date'].astype(str) + " " + df["Time"]
    df['HourUTC'] = df['HourUTC'].apply(lambda x: str(x.replace('.', '-')))
    df['HourUTC'] = pd.to_datetime(df['HourUTC'], format='%d-%m-%Y %H:%M')
    df = df[['HourUTC', 'Actual']]
    df['HourUTC'] = df['HourUTC'].dt.tz_localize(None)
    df = df.drop_duplicates('HourUTC', keep='first')
    df.set_index('HourUTC', inplace=True)

    ### Sort out DST issues with duplicate values and missing observation (2022-01-01 00:00)
    df1Jan = np.mean([df.loc['2022-01-01 01:00']['Actual'], df.loc['2021-12-31 23:00']['Actual']])
    line = pd.DataFrame({"Actual": df1Jan}, index=[datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')])
    df = df.append(line, ignore_index=False)
    df = df.sort_index()
    df = df.iloc[0:len(df['Actual']) - 1, :]
    """ Power production is already normalized """

    """ Load data files from folder """
    temp_dir = os.path.join(cwd, 'ClimateData/data.csv')
    data = pd.read_csv(temp_dir)
    data.set_index('HourUTC', inplace=True)

    # %% Step 2: Creating the training and test datasets
    wind_cubed = np.array((data['wind_speed [m/s]'] ** 3).values)
    windXpressure = np.array((data['wind_speed [m/s]'] ** 3).values * data['pressure [hPa]'].values)
    data['wind_cubed'] = wind_cubed
    data['wind_energy'] = windXpressure
    data['production'] = df['Actual'].values
    data['past_prod'] = data['production'].shift(periods = 24, fill_value=np.mean(data['production'].iloc[0:24]))

    # %% Standardization
    attributeNames = np.asarray(data.columns)
    # Not standardizing production
    attributeNames = np.delete(attributeNames,[6,7])
    dfs = data.copy()
    mu_dfs = np.mean(dfs[attributeNames])
    std_dfs = np.std(dfs[attributeNames])

    for i in range(0, len(attributeNames)):
        dfs[attributeNames[i]] = (dfs[attributeNames[i]] - mu_dfs[i]) / std_dfs[i]

    dfs['ones'] = 1
    
    return dfs, mu_dfs, std_dfs


def readRegPrice(file):
    cwd = os.getcwd()
    f = os.path.join(cwd,file)
    df = pd.read_csv(f)
    target_value = '"'
    df = df.stack().str.replace(target_value,'').unstack()
    column_mapping = {col: col.replace('"', '') for col in df.columns}
    df.rename(columns=column_mapping, inplace=True)
    df[df.columns[0].split(",")] = df[df.columns[0]].str.split(",",expand=True)
    df = df[df.columns[-1]]
    return df


def getPrices():
    # Regulation prices:
    df_up21 = readRegPrice('Up-regulation price_2021.csv')
    df_up22 = readRegPrice('Up-regulation price_2022.csv')
    df_down21 = readRegPrice('Down-regulation price_2021.csv')
    df_down22 = readRegPrice('Down-regulation price_2022.csv')
    up = np.append(df_up21.values,df_up22.values)
    down = np.append(df_down21.values,df_down22.values)
    
    #Day ahead prices:
    cwd = os.getcwd()
    f_ahead = os.path.join(cwd,'Day-ahead price.xlsx')
    df_ahead = pd.read_excel(f_ahead)
    df_ = df_ahead.loc[df_ahead['PriceArea'] == 'DK2'].reset_index(drop = True)
    df_ = df_.loc[df_['HourUTC'] >= '2021']
    df_ = df_.loc[df_['HourUTC'] < '2023']
    df_ = df_[df_.columns[-1]]
    
    df = pd.DataFrame()
    df['Spot'] = df_
    df['Up'] = up
    df['Down'] = down
    
    return df


def loadBids():
    cwd = os.getcwd()
    """ Load actual wind power from cwd """
    temp_dir = os.path.join(cwd, 'optimal bids.csv')
    df = pd.read_csv(temp_dir)
    
    return np.array(df['Opt-Bid'].values)


if __name__ == "__main__":
    from regression import cf_fit, cf_predict
    
    data = prepData()
    
    # %% Step 3: First sample data
    sizes = [100, 1000, len(data)]
    feat = [['ones', 'wind_speed [m/s]'], ['ones', 'wind_speed [m/s]', 'temperature [C]'], ['ones', 'wind_cubed'],
            ['ones', 'temperature [C]']]
    mse_list = []
    
    y = np.array(data['production'])
    for features in feat:
        mse_list.append(features)
        for size in sizes:
            X = np.array(data[features])
            # X = np.matrix([x1,data[cols[0]],data[cols[1]],data[cols[2]]]).T
            X_slice = X[0:size]
            y_slice = y[0:size]
    
            X_train, X_test, y_train, y_test = train_test_split(X_slice, y_slice, test_size=0.4, shuffle=False)
    
            # Closed form linear regression:
            beta = cf_fit(X_train, y_train)
            y_pred = cf_predict(beta, X_test)
            mse = mean_squared_error(y_test, y_pred)  # 0.028742528161411984
            mse_list.append(mse)
    print(mse_list)
    
    '''
    #%% Step 4: Non-linear regression
    wind_cubed = np.array((data['wind_speed [m/s]']**3).values)
    windXpressure = np.array((data['wind_speed [m/s]']**3).values*data['pressure [hPa]'].values)
    dfs['wind_cubed'] = wind_cubed
    dfs['wind_energy'] = windXpressure
    '''
    
