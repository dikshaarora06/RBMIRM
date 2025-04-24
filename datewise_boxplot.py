import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

# Read the output and target files into pandas dataframes
output_df_t = pd.read_excel('/home/diksha/Documents/output_t.xlsx')             #change name according to phase (test or train)
target_df_t = pd.read_excel('/home/diksha/Documents/target_t.xlsx')
output_df = pd.read_excel('/home/diksha/Documents/output.xlsx')             #change name according to phase (test or train)
target_df = pd.read_excel('/home/diksha/Documents/target.xlsx')

# Group the data by date
output_grouped_t = output_df_t.groupby(output_df_t.iloc[:, 0])
target_grouped_t = target_df_t.groupby(target_df_t.iloc[:, 0])
# Group the data by date
output_grouped = output_df.groupby(output_df.iloc[:, 0])
target_grouped = target_df.groupby(target_df.iloc[:, 0])

# Create a list to store the data for each date
data_list = []
dates = []

# Iterate over each group (date)
for date, output_data in output_grouped:
    target_data = target_grouped.get_group(date)
    dates.append(date)
    # Get the parameter names from the output data columns (excluding the first two columns)
    parameters = output_data.columns[2:]
    # Create tensors for output and target data
    output_tensor = torch.tensor(output_data[parameters].values, dtype=torch.float32)
    target_tensor = torch.tensor(target_data[parameters].values, dtype=torch.float32)
    loss = abs(output_tensor - target_tensor)
    
    # Append the loss data for the current date to the data_list
    data_list.append(loss.detach().cpu().numpy())

    # Save loss data to a CSV file
    loss_df = pd.DataFrame(loss.detach().cpu().numpy(), columns=parameters)
    loss_df.to_csv(f'lossnewd_{date}.csv', index=False)
    


# Create a list to store the data for each date
data_list_t = []
dates_t = []

# Iterate over each group (date)
for date_t, output_data_t in output_grouped_t:
    target_data_t = target_grouped_t.get_group(date_t)
    dates_t.append(date_t)
    # Get the parameter names from the output data columns (excluding the first two columns)
    parameters_t = output_data_t.columns[2:]
    # Create tensors for output and target data
    output_tensor_t = torch.tensor(output_data_t[parameters_t].values, dtype=torch.float32)
    target_tensor_t = torch.tensor(target_data_t[parameters_t].values, dtype=torch.float32)
    loss_t = abs(output_tensor_t - target_tensor_t)
    
    # Append the loss data for the current date to the data_list
    data_list_t.append(loss_t.detach().cpu().numpy())

    # Save loss data to a CSV file
    loss_df_t = pd.DataFrame(loss_t.detach().cpu().numpy(), columns=parameters_t)
    loss_df_t.to_csv(f'lossnewd_test_{date_t}.csv', index=False)
#now after this run d.py