import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('tips.csv')

# Create a 2x2 subplot layout
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram of total bill
axs[0, 0].hist(df['total_bill'], bins=20, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Histogram of Total Bill')
axs[0, 0].set_xlabel('Total Bill')
axs[0, 0].set_ylabel('Frequency')

# 2. Boxplot of tip amounts
axs[0, 1].boxplot(df['tip'], vert=True)
axs[0, 1].set_title('Boxplot of Tips')
axs[0, 1].set_ylabel('Tip Amount')

# 3. Pie chart showing gender distribution
gender_counts = df['sex'].value_counts()
axs[1, 0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
axs[1, 0].set_title('Gender Distribution')

# 4. Bar chart of average total bill by day
avg_total_by_day = df.groupby('day')['total_bill'].mean()
axs[1, 1].bar(avg_total_by_day.index, avg_total_by_day.values, color='lightgreen')
axs[1, 1].set_title('Average Total Bill by Day')
axs[1, 1].set_xlabel('Day')
axs[1, 1].set_ylabel('Average Total Bill')

# Adjust layout
plt.tight_layout()
plt.show()
