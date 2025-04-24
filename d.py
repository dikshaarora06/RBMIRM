import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have four CSV files named loss_date1.csv, loss_date2.csv, loss_date3.csv, and loss_date4.csv
# Load data from CSV files
file_paths = ['lossnewd_12-8-22.csv', 'lossnewd_7-9-22.csv', 'lossnewd_20-9-22.csv', 'lossnewd_10-10-22.csv']
file_paths_test = ['lossnewd_test_12-8-22.csv', 'lossnewd_test_7-9-22.csv', 'lossnewd_test_20-9-22.csv', 'lossnewd_test_10-10-22.csv']

dfs = [pd.read_csv(file) for file in file_paths]
dfs_test = [pd.read_csv(file) for file in file_paths_test]

# Extracting dates from training file names
dates = [file.split('_')[1].split('.')[0] for file in file_paths]

# Extracting dates from testing file names
dates_test = [file.split('_')[2].split('.')[0] for file in file_paths_test]

# Extract the parameters (including the first column, which is the date)
param_names = dfs[0].columns.tolist()
params_group1 = param_names[:3]
param_names_test = dfs_test[0].columns.tolist()
params_group1_test = param_names_test[:3]

# Combine all data into a single DataFrame with an additional 'Date' column
dfs_combined = []
for i, df in enumerate(dfs):
    df['Date'] = dates[i]
    dfs_combined.append(df)
combined_data = pd.concat(dfs_combined)
# Melt the DataFrame to have a single 'value' column for all parameters
melted_data = pd.melt(combined_data, id_vars=['Date'], value_vars=param_names, var_name='Parameter', value_name='Value')
# Filter the data to ignore outliers outside the range 0-1
melted_data = melted_data[(melted_data['Value'] >= 0) & (melted_data['Value'] <= 1)]
# Calculate the median loss error values datewise for all parameters
mean_values = melted_data.groupby(['Date', 'Parameter'])['Value'].mean().unstack()
# Round the median values to two decimal places
mean_values = mean_values.round(2)
# Print the table of median values
#print("Mean Loss Error Values Datewise testing (Rounded to Two Decimal Places):")
print("Mean Loss Error Values Datewise Training (Rounded to Two Decimal Places):")
print(mean_values)
# Plotting Grouped Boxplots for the first 4 parameters
fig1, ax1 = plt.subplots(figsize=(6, 6))
sns.boxplot(x='Parameter', y='Value', hue='Date', data=melted_data[melted_data['Parameter'].isin(melted_data['Parameter'].unique()[:11])], ax=ax1)
#ax1.set_title('Parameters 1 to 4')
ax1.set_xlabel('')  # Remove x label
ax1.set_ylabel('')  # Remove y label
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc='upper center')
#fig1.savefig('d_k100_test.jpeg', dpi=300)
fig1.savefig('d_new100_k.jpeg', dpi=300)
# Adjust the space between the subplots
plt.tight_layout()
# Show the plot
plt.show()

dfs_combined_test = []
for i, df in enumerate(dfs_test):
    df['Date'] = dates[i]
    dfs_combined_test.append(df)
combined_data_test = pd.concat(dfs_combined_test)
# Melt the DataFrame to have a single 'value' column for all parameters
melted_data_test = pd.melt(combined_data_test, id_vars=['Date'], value_vars=param_names_test, var_name='Parameter', value_name='Value')
# Filter the data to ignore outliers outside the range 0-1
melted_data_test = melted_data_test[(melted_data_test['Value'] >= 0) & (melted_data_test['Value'] <= 1)]
# Calculate the median loss error values datewise for all parameters
mean_values_test = melted_data_test.groupby(['Date', 'Parameter'])['Value'].mean().unstack()
# Round the median values to two decimal places
mean_values_test = mean_values_test.round(2)
# Print the table of median values
#print("Mean Loss Error Values Datewise testing (Rounded to Two Decimal Places):")
print("Mean Loss Error Values Datewise Testing (Rounded to Two Decimal Places):")
print(mean_values_test)
# Plotting Grouped Boxplots for the first 4 parameters
fig2, ax2 = plt.subplots(figsize=(6, 6))
sns.boxplot(x='Parameter', y='Value', hue='Date', data=melted_data_test[melted_data_test['Parameter'].isin(melted_data_test['Parameter'].unique()[:11])], ax=ax2)
ax2.set_xlabel('')  # Remove x label
ax2.set_ylabel('')  # Remove y label
ax2.tick_params(axis='x', rotation=45)
ax2.legend(loc='upper center')
fig2.savefig('d_k100_test.jpeg', dpi=300)

# Adjust the space between the subplots
plt.tight_layout()
# Show the plot
plt.show()