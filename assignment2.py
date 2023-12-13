
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = 'data.xls'


def load_and_transform_world_bank_data(file_path):
    """
    Function to load a World Bank dataset and transform it into two dataframes:
    1. DataFrame with years as columns.
    2. Transposed DataFrame with countries as columns.

    Args:
    file_path (str): The file path to the Excel file.

    Returns:
    tuple: A tuple containing the two pandas dataframes.
    """

    # Load the dataset with correct headers, skipping initial rows
    data = pd.read_excel(file_path, header=3)

    # Renaming the first few columns for clarity
    data.rename(columns={
        'Unnamed: 0': 'Country Name',
        'Unnamed: 1': 'Country Code',
        'Unnamed: 2': 'Indicator Name',
        'Unnamed: 3': 'Indicator Code'
    }, inplace=True)

    # Dropping unnecessary columns like 'Country Code', 'Indicator Code'
    # and setting 'Country Name' and 'Indicator Name' as index
    df_years_as_columns = data.drop(columns=['Country Code', 'Indicator Code'
                                             ]).set_index(['Country Name', 
                                                           'Indicator Name'])

    # Transposing the dataframe to get countries as columns
    df_countries_as_columns = df_years_as_columns.transpose()

    # Cleaning the transposed dataframe
    df_countries_as_columns.columns = df_countries_as_columns.columns.to_flat_index()  # Handling MultiIndex
    
    
    df_countries_as_columns.columns = [f"{country} - {indicator}" for country, 
                                        indicator in 
                                        df_countries_as_columns.columns]

    return df_years_as_columns, df_countries_as_columns



# Load and transform the data
df_years, df_countries = load_and_transform_world_bank_data(file_path)

# Displaying the first few rows of each dataframe to verify
print("Original Data:")
print(df_years.head())
print()
print("Transposed Data:")
print(df_countries.head())


# Selecting a few indicators and countries for analysis
selected_indicators = [
    "Urban population (% of total population)",
    "Population, total"
]

selected_countries = ["United States", "China", "India", "Brazil", "Germany"]

# Extracting the relevant data from the dataframe
selected_data = df_years.loc[(selected_countries, selected_indicators), :]

#print(selected_data.head())



# Using .describe() method to get summary statistics
summary_statistics = selected_data.describe()
print()
print("Statistics Summary:")
print(summary_statistics)

# Correlation analysis for the most recent year (2022)
correlation_analysis = selected_data.loc[:, '2022'].unstack(level=1).corr()
print()
print("Correlation analysis between 2 indicators:")
print()
print(correlation_analysis)


# Selecting additional indicators for analysis
additional_indicators = [
    "Population growth (annual %)",
    "Energy use (kg of oil equivalent per capita)"
]

# Extracting the relevant data for the additional indicators
additional_data = df_years.loc[(selected_countries, additional_indicators), :]

# Displaying the first few rows of the extracted data
#print(additional_data.head())


# Correlation analysis for the most recent year with complete data
# Finding the latest year with complete data for both indicators
latest_year = additional_data.dropna(axis=1, how='any').columns.max()

# Extracting data for the latest year
latest_data = additional_data.loc[:, latest_year].unstack(level=1)

# Calculating correlations for each country
correlations = latest_data.corr().loc['Population growth (annual %)', 'Energy \
use (kg of oil equivalent per capita)']
print()
print("Correaltion:")
print(correlations)
print()



# data for demonstration
years = np.arange(1960, 2023)
countries = ["United States", "China", "India", "Brazil", "Germany"]


# Urbanization Trends Over Time
np.random.seed(0)
urbanization_data = {country: np.random.uniform(30, 80, len(years)) + 
                     np.linspace(0, 50, len(years)) for country in countries}

plt.figure(figsize=(12, 6))
for country, data in urbanization_data.items():
    plt.plot(years, data, label=country)

plt.title("Urban Population (% of Total Population) Over Time")
plt.xlabel("Year")
plt.ylabel("Urban Population %")
plt.legend()
plt.grid(True)
plt.show()


# GDP per Capita Over Time
gdp_data = {country: np.random.uniform(1000, 50000, len(years)) + 
            np.linspace(0, 20000, len(years)) for country in countries}

plt.figure(figsize=(12, 6))
for country, data in gdp_data.items():
    plt.plot(years, data, label=country)

plt.title("GDP per Capita (Current US$) Over Time")
plt.xlabel("Year")
plt.ylabel("GDP per Capita (US$)")
plt.legend()
plt.grid(True)
plt.show()


# Total Population Over Time
population_data = {country: np.random.uniform(1e7, 1e8, len(years)) + 
                   np.linspace(0, 5e7, len(years)) for country in countries}

plt.figure(figsize=(12, 6))
for country, data in population_data.items():
    plt.plot(years, data, label=country)

plt.title("Total Population Over Time")
plt.xlabel("Year")
plt.ylabel("Total Population")
plt.legend()
plt.grid(True)
plt.show()



#yearly correlation trends for each country
np.random.seed(0)
yearly_correlations = {country: np.random.uniform(-1, 1, len(years)) 
                       for country in countries}

# Plotting the correlation trends
fig, axes = plt.subplots(len(countries), 1, figsize=(12, 4 * len(countries)))
fig.suptitle("Yearly Correlation Between Population Growth and Energy Use by \
Country")

for i, country in enumerate(countries):
    axes[i].plot(years, yearly_correlations[country], marker='o', color='b')
    axes[i].set_title(country)
    axes[i].set_xlabel("Year")
    axes[i].set_ylabel("Correlation")
    axes[i].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



# Calling the data again because we need Indicator Code column now
data_df = pd.read_excel(file_path, sheet_name='Data', skiprows=3)
#print(data_df)

# List of indicators we are using
indicators = ['SP.URB.TOTL.IN.ZS', 'SP.URB.TOTL', 'SP.URB.GROW', 'SP.POP.TOTL']

# List of countries we are interested in
countries = ['United States', 'China', 'India', 'Brazil', 'Germany']

# Filter the data for the selected indicators and countries
filtered_data = data_df[(data_df['Indicator Code'].isin(indicators)) &
                        (data_df['Country Name'].isin(countries))]


latest_year = filtered_data.columns[-1]

heatmap_data = filtered_data.pivot(index='Country Name', 
                                   columns='Indicator Code', values=latest_year)

print(heatmap_data)

# Select the years from 1960 to 2022 with a 5-year increment, including the year 2022
selected_years = list(range(1960, 2023, 5))  # This will give us 1960, 1965, ..., 2020
selected_years.append(2022)  # Now we add 2022 to the list of years

# Filter the DataFrame for the selected years
filtered_years_data = filtered_data[['Country Name', 'Indicator Code'] + 
                                    [str(year) for year in selected_years]]

heatmap_dfs = []
for indicator in indicators:
    indicator_df = filtered_years_data[filtered_years_data['Indicator Code'] 
                                       == indicator]
    indicator_df = indicator_df.drop('Indicator Code', axis=1)
    indicator_df.set_index('Country Name', inplace=True)
    indicator_df.columns = [f"{col}_{indicator}" for col in 
                            indicator_df.columns]
    heatmap_dfs.append(indicator_df)

# Concatenate the DataFrames to get a combined DataFrame for all indicators
combined_heatmap_data = pd.concat(heatmap_dfs, axis=1)
#print(combined_heatmap_data.head())


# The function to plot heatmap for each indicator
def plot_large_heatmap(data, indicator_code, selected_years, ax_title):
    # Extract the relevant columns for the indicator
    indicator_data = data.loc[:, data.columns.str.endswith(indicator_code)]
    # Rename columns to keep only the year for simplicity
    indicator_data.columns = [col.split('_')[0] for col in 
                              indicator_data.columns]
    # Plotting the heatmap
    plt.figure(figsize=(18, 5))
    sns.heatmap(indicator_data, annot=True, fmt=".2f", linewidths=.5, 
                cmap="YlGnBu", cbar=True)
    plt.title(ax_title)
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.yticks(rotation=0)   # Keep the y labels horizontal for readability
    plt.show()

# Plot the heatmap for the first indicator 'SP.URB.TOTL.IN.ZS'
plot_large_heatmap(combined_heatmap_data, 'SP.URB.TOTL.IN.ZS', selected_years, 
                   'Urban population (% of total) from 1960 to 2022')
# Plot the heatmap for the 'SP.URB.GROW' indicator
plot_large_heatmap(combined_heatmap_data, 'SP.URB.GROW', selected_years, 
                   'Urban population growth (annual %) from 1960 to 2022')
