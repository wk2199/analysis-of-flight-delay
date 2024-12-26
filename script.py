# %%

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
import statsmodels.api as sm

# %%

flight_data = pd.read_csv('lax_to_jfk.csv', skiprows = 0).fillna(0)

# %%

flight_data.head()

# %%

sorted_column_names = sorted(flight_data)
print(sorted_column_names)

# %%

flight_cont = flight_data[['ArrDelay','ArrDelayMinutes','CarrierDelay','DepDelay','DepDelayMinutes','SecurityDelay', 'WeatherDelay']]
round(flight_cont.describe(),2)

# %%

flight_data["FlightDate"] = pd.to_datetime(flight_data["FlightDate"])

flight_data["Year"] = flight_data["FlightDate"].dt.year

plt.bar('Year', 'ArrDelayMinutes', data=flight_data)
plt.xlabel('Year')
plt.ylabel('Arrival Delay (Minutes)')
plt.title('Mean Arrival Delay (minutes) over the Years')
plt.show()

# %% [markdown]

# %%

plt.figure(figsize=(10, 6))  
sns.barplot(x='Reporting_Airline', y='ArrDelayMinutes', data=flight_data)
plt.xlabel('Reporting Airline')
plt.ylabel('Average Arrival Delay (minutes)')
plt.title('Mean Arrival Delay by each Reporting Airline')
plt.show()

# %% [markdown]

# %%


flight_data['DepTime']=flight_data['DepTime'].apply(str)

flight_data['DepTime'] = flight_data['DepTime'].str.strip()
flight_data['DepTime'] = flight_data['DepTime'].str.pad(width=4, side='left', fillchar='0')

flight_data["DateTime"] = pd.to_datetime(flight_data["DepTime"], format="%H%M")

flight_data = flight_data.set_index("DateTime")
flight_data["DepartureTime"] = flight_data.index.time
flight_data["DepartureTime"] = pd.to_datetime(flight_data["DepartureTime"], format="%H:%M:%S")
flight_data["DepartureTime"] = mdates.date2num(flight_data["DepartureTime"])
plt.figure(figsize = (10, 7))
plt.bar('DepartureTime', 'CarrierDelay', data=flight_data, width=0.01)
plt.xlabel("Departure hour")
plt.ylabel("Average Arrival delay (minutes)")
plt.title("Average Carrier Delay by Departure Hour")
plt.tight_layout()

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.show()

# %% [markdown]

# %%

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
flight_data['Month'] = pd.Categorical(flight_data['Month'], categories=range(1, 13), ordered=True)
flight_data['Month'] = flight_data['Month'].cat.rename_categories(month_names)

flight_data.sort_values('Month', inplace=True)

plt.figure(figsize = (10, 7))
plt.bar('Month', 'WeatherDelay', data=flight_data)
plt.xlabel("Month of the Year")
plt.ylabel("Average weather delay (minutes)")
plt.title("Average Weather Delay by Month of the Year")
plt.show()

# %%

plt.figure(figsize=(10, 6))  
sns.barplot(x='DayOfWeek', y='ArrDelayMinutes', data=flight_data, ci = False)
plt.xlabel('Day of the Week')
plt.ylabel('Average Arrival Delay (minutes)')
plt.title('Mean Arrival Delay by each day of the week')
plt.show()

# %% [markdown]

# %%

flight_data['prvious_delay'] = flight_data['DepDelay'].shift(1)

plt.figure(figsize = (10, 7))
plt.bar('DepartureTime', 'DepDelay', data=flight_data, label = "Departure delay", width=0.01)
plt.bar('DepartureTime', 'prvious_delay', data=flight_data, label = "Previous departure Delay", width=0.01)
plt.xlabel("Departure hour")
plt.ylabel("Average delay (minutes)")
plt.title("Departure and previous departure delay by Departure Hour")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.legend()
plt.show()

# %%
def map_to_quarter(month):
    if month in [1, 2, 3]:
        return 'Q1'
    elif month in [4, 5, 6]:
        return 'Q2'
    elif month in [7, 8, 9]:
        return 'Q3'
    elif month in [10, 11, 12]:
        return 'Q4'
    else:
        return 'Unknown'  

flight_data['Quarter'] = flight_data['Month'].apply(map_to_quarter)

# %%
def map_to_b(DayOfWeek):
    if DayOfWeek in [1, 5, 7]:
        return ' Very busy'
    elif DayOfWeek in [2,3]:
        return 'less busy'
    elif DayOfWeek in [4,6]:
        return 'moderately busy'
    else:
        return 'Unknown'  

flight_data['Busy'] = flight_data['DayOfWeek'].apply(map_to_b)

# %%


flight_data['DepTime'] = flight_data['DepTime'].apply(int)

Day_dummy = pd.get_dummies(flight_data['Busy']).astype(int)

Month_dummy = pd.get_dummies(flight_data['Quarter']).astype(int)

flight_data = pd.concat([flight_data, Day_dummy], axis=1)
flight_data = pd.concat([flight_data, Month_dummy], axis=1)

y = flight_data['ArrDelayMinutes']  
X = flight_data.iloc[:, [18]+[8]+[15]+[16] + list(range(24,29))]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()


print(model.summary())

# %%
selected_columns = flight_data[['WeatherDelay', 'DepTime', 'CarrierDelay', 'SecurityDelay', ]]

correlation_matrix = selected_columns.corr(method='spearman')

print(correlation_matrix)

plt.figure(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

plt.show()

# %%



