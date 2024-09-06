# Sales Data Analysis

## Overview
This analysis explores meal sales data to derive insights through various visualizations. The dataset contains details about meal prices, types, participants, and meal dates.

## Dataset Source and Description
The dataset, Invoices.csv, is loaded using Pandas for analysis. It contains various details about meal transactions such as Date, Type of Meal, Meal Price, Participants, and Date of Meal.

## Columns in the dataset:

Date: The date when the meal transaction was recorded.
Date of Meal: The date when the meal was served.
Type of Meal: The category of the meal (e.g., Breakfast, Lunch).
Meal Price: The total price for each meal transaction.
Participants: List of people involved in the meal, formatted as strings of names inside brackets.
## Data Preprocessing

 Date Conversion: Both Date and Date of Meal were converted to datetime format for better time-based analysis.
Handling Participants Data: The Participants column was cleaned by removing brackets and splitting the participants into separate rows for individual analysis.
## Exploratory Data Analysis (EDA)
The Exploratory Data Analysis (EDA) focuses on uncovering patterns, trends, and relationships within the dataset using various visualizations and statistical summaries.

## 1. Total Sales
Type: Textual Summary
Description: Calculates the total sales from all meal transactions.

Code:

```python
Copy code
total_sales = df['Meal Price'].sum()
print(f"Total Sales: ${total_sales}")
```
## 2. Sales by Type of Meal
Type: Bar Chart
Description: Displays the total sales for each type of meal.

Code:

```python
Copy code
sales_by_meal_type = df.groupby('Type of Meal')['Meal Price'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
colors = sns.color_palette("coolwarm", len(sales_by_meal_type))
sales_by_meal_type.plot(kind='bar', color=colors)
plt.title('Sales by Type of Meal', fontsize=15, fontweight='bold')
plt.xlabel('Type of Meal')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()
```
## 3. Top 5 Most Frequent Participants
Type: Bar Chart
Description: Shows the top 5 participants who appeared most frequently in the meal transactions.

Code:

```python
Copy code
# Count the occurrences of each participant
participant_counts = participants_expanded.value_counts()

# Get the top 5 most frequent participants
top_5_participants = participant_counts.head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_participants.values, y=top_5_participants.index, palette="coolwarm")
plt.title('Top 5 Most Frequent Participants')
plt.xlabel('Number of Meals')
plt.ylabel('Participants')
plt.tight_layout()
plt.show()
```
## 4. Monthly Sales Trend
Type: Line Plot
Description: Displays the total sales trend over time, grouped by month.

Code:

```python
Copy code
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales_trend = df.groupby('Month')['Meal Price'].sum()
plt.figure(figsize=(10, 6))
monthly_sales_trend.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.tight_layout()
plt.show()
```
## 5. Box Plot of Meal Prices by Type of Meal
Type: Box Plot
Description: Visualizes the distribution of meal prices for each type of meal.

Code:

```python
Copy code
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type of Meal', y='Meal Price', data=df, palette="Set3")
plt.title('Box Plot of Meal Prices by Type of Meal', fontsize=15, fontweight='bold')
plt.xlabel('Type of Meal')
plt.ylabel('Meal Price')
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()
```
## 6. Heatmap of Meal Prices by Meal Type and Company
Type: Heatmap
Description: Displays a pivot table of total meal prices grouped by Type of Meal and Company Id.

Code:

```python
Copy code
meal_price_pivot = df.pivot_table(index='Type of Meal', columns='Company Id', values='Meal Price', aggfunc='sum')
plt.figure(figsize=(14, 7))
sns.heatmap(meal_price_pivot, cmap='YlGnBu', annot=True, fmt='.0f')
plt.title('Heatmap of Meal Prices by Meal Type and Company')
plt.xlabel('Company Id')
plt.ylabel('Type of Meal')
plt.tight_layout()
plt.show()
```
## 7. Distribution of Meal Prices
Type: Histogram
Description: Displays the distribution of meal prices.

Code:

```python
Copy code
plt.figure(figsize=(10, 6))
sns.histplot(df['Meal Price'], bins=30, kde=True, color='purple')
plt.title('Distribution of Meal Prices', fontsize=15, fontweight='bold')
plt.xlabel('Meal Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```
## 8. Scatter Plot of Meal Prices Over Time
Type: Scatter Plot
Description: Displays meal prices over time for each transaction.

Code:

```python
Copy code
plt.figure(figsize=(12, 6))
plt.scatter(df['Date of Meal'], df['Meal Price'], alpha=0.5, color='red')
plt.title('Scatter Plot of Meal Prices Over Time', fontsize=15, fontweight='bold')
plt.xlabel('Date of Meal')
plt.ylabel('Meal Price')
plt.tight_layout()
plt.show()
```
## Result
This analysis successfully provides a detailed exploration of the meal sales data. Key insights are derived from customer behavior, meal price trends, and sales performance over time. For a detailed view, refer to the attached Jupyter Notebook for additional visualizations and analysis.
