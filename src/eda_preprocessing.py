import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the dataset
df = pd.read_csv('data/complaint.csv')

# Initial EDA
print("Dataset Head:\n", df.head())
print("\nDataset Info:\n", df.info())
print("\nDataset Summary:\n", df.describe())

# Analyze distribution of complaints by product
product_counts = df['Product'].value_counts()
print("\nComplaints by Product:\n", product_counts)
product_counts.plot(kind='bar')
plt.title('Distribution of Complaints by Product')
plt.xlabel('Product')
plt.ylabel('Number of Complaints')
plt.savefig('notebooks/product_distribution.png')
plt.close()

# Calculate and visualize narrative lengths
df['narrative_length'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))
plt.hist(df['narrative_length'], bins=50)
plt.title('Distribution of Narrative Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.savefig('notebooks/narrative_length_distribution.png')
plt.close()
print("\nNarrative Length Stats:\n", df['narrative_length'].describe())

# Count complaints with and without narratives
with_narrative = df['Consumer complaint narrative'].notna().sum()
without_narrative = df['Consumer complaint narrative'].isna().sum()
print(f"\nComplaints with narratives: {with_narrative}")
print(f"Complaints without narratives: {without_narrative}")

# Filter dataset
specified_products = ['Credit card', 'Personal loan', 'Buy Now, Pay Later (BNPL)', 'Savings account', 'Money transfers']
filtered_df = df[df['Product'].isin(specified_products) & df['Consumer complaint narrative'].notna()]
print("\nFiltered Complaints by Product:\n", filtered_df['Product'].value_counts())
print("Missing narratives in filtered data:", filtered_df['Consumer complaint narrative'].isna().sum())

# Clean narratives
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text)

# Save filtered dataset
filtered_df.to_csv('data/filtered_complaints.csv', index=False)
print("\nFiltered dataset saved to 'data/filtered_complaints.csv'")
