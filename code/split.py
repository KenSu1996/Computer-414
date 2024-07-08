import pandas as pd
folders = ["black_males", "indian_females", "white_females", "black_females", "white_males", "asian_males", "indian_males", "asian_females"]

# Replace 'your_csv_file.csv' with the path to your CSV file
csv_file_path = './bfw-v0.1.5-datatable.csv'
# Replace 'category_column_name' with the actual name of your first column
category_column_name = 'att1'

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file_path)

# Ensure that the category column is a string to match with the folder names
df[category_column_name] = df[category_column_name].astype(str)

# Initialize an empty list to store the DataFrame slices
filtered_data = []

# Iterate over each category and get a sample
for folder in folders:
    # Filter the DataFrame by the current category
    category_data = df[df[category_column_name] == folder]
    
    # Sample 100 rows from the category data
    sample_data = category_data.sample(n=min(1000, len(category_data)), random_state=1)
    
    # Append the sample to the list
    filtered_data.append(sample_data)

# Concatenate all the samples into a single DataFrame
result_df = pd.concat(filtered_data)

# Save the new DataFrame to a new CSV file
result_df.to_csv('filtered_data.csv', index=False)
