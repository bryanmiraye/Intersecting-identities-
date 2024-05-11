Hypothesis: Intersecting identities, specifically combinations of race, gender, and socioeconomic status, significantly influence an individual’s or group’s vulnerability to food insecurity and economic instability. It is anticipated that marginalized groups—such as women of color from lower socioeconomic backgrounds—will experience higher rates of food insecurity and economic hardship compared to their counterparts from higher socioeconomic statuses or with more socially privileged identities (e.g., male, white). This hypothesis is based on the premise that intersectionality compounds vulnerability, where each additional marginal identity amplifies the risk of experiencing economic difficulties and limited access to food resources.”

import pandas as pd de
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from google.colab import drive
drive.mount('/content/drive')

# Food Security Dataset
path1 = "/content/drive/MyDrive/CSV Files/dec22pub (1).csv"

df1 = pd.read_csv(path1)
df1 = pd.read_csv(path1, low_memory=False)
df1.drop_duplicates(subset='HRHHID').shape
df1.head()



# Basic Monthly CPS
path2 = "/content/drive/MyDrive/CSV Files/dec22pub.csv"

df2 = pd.read_csv(path2)

merge1 = df1.merge(df2, left_on='QSTNUM', right_on='qstnum')

# Annual social and economic supplement for HouseHold
path3 = "/content/drive/MyDrive/CSV Files/hhpub22.csv"

df3 = pd.read_csv(path3)
df3['GEDIV'] = range(1, len(df3) + 1)
df3.drop_duplicates(subset='H_IDNUM').shape

merge2=merge1.merge(df3, left_on='qstnum', right_on='GEDIV')


necessary_columns = ['HEFAMINC_x', 'PTDTRACE', 'PESEX']

# removing rows with NaN or negative values
merge2_clean = merge2.dropna(subset=['HEFAMINC_x'])[necessary_columns]  # Drop rows where 'HEFAMINC_x' is NaN and select columns
merge2_clean = merge2_clean[merge2_clean['HEFAMINC_x'] >= 0]  # Keep rows with 'HEFAMINC_x' >= 0

merge2_clean = merge2_clean.copy()

# Map for PTDTRACE (Race)
race_map = {
    1: "White Only", 2: "Black Only", 3: "American Indian, Alaskan Native Only",
    4: "Asian Only", 5: "Hawaiian/Pacific Islander Only", 6: "White-Black",

}

# Map for PESEX (Gender)
sex_map = {
    1: "Male", 2: "Female"
}

merge2_clean['Race'] = merge2_clean['PTDTRACE'].map(race_map)
merge2_clean['Gender'] = merge2_clean['PESEX'].map(sex_map)

# Income bracket mapping
income_bracket_map = {
    1: "Less than $5,000", 2: "$5,000 to $7,499", 3: "$7,500 to $9,999",
    4: "$10,000 to $12,499", 5: "$12,500 to $14,999", 6: "$15,000 to $19,999",
    7: "$20,000 to $24,999", 8: "$25,000 to $29,999", 9: "$30,000 to $34,999",
    10: "$35,000 to $39,999", 11: "$40,000 to $49,999", 12: "$50,000 to $59,999",
    13: "$60,000 to $74,999", 14: "$75,000 to $99,999", 15: "$100,000 to $149,999",
    16: "$150,000 or more"
}

# Apply income bracket mapping
merge2_clean['Income Bracket'] = merge2_clean['HEFAMINC_x'].map(income_bracket_map)

# Group by Gender and Race, then calculate mean income for each group
grouped_income = merge2_clean.groupby(['Gender', 'Race'])['HEFAMINC_x'].mean().reset_index()

# Sort by mean income to easily identify highest and lowest incomes
grouped_income_sorted = grouped_income.sort_values(by='HEFAMINC_x', ascending=False)

# Highest income groups
highest_income_groups = grouped_income_sorted.head()

# Lowest income groups
lowest_income_groups = grouped_income_sorted.tail()


print("Groups with the Highest Average Family Incomes:")
print(highest_income_groups)

print("\nGroups with the Lowest Average Family Incomes:")
print(lowest_income_groups)


plt.figure(figsize=(12, 6))

# Iterate through each gender and race group
for index, row in grouped_income.iterrows():
    gender_race = f"{row['Gender']} - {row['Race']}"
    plt.plot(gender_race, row['HEFAMINC_x'], marker='o', linestyle='-')

plt.title('Average Family Income by Gender and Race')
plt.xlabel('Gender - Race')
plt.ylabel('Average Family Income')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()

plt.show()



sns.barplot(data=grouped_income, x='Race', y='HEFAMINC_x', hue='Gender')
plt.title('Average Family Income by Race and Gender')
plt.xticks(rotation=45)
plt.ylabel('Average Income')
plt.xlabel('Race')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()



# Replace negative values with NaN for the variables of interest
variables_of_interest = ['HRFS12MC', 'HRFS12M8', 'PESEX', 'PTDTRACE']
merge2[variables_of_interest] = merge2[variables_of_interest].applymap(lambda x: np.nan if x < 0 else x)

gender_map = {1: 'Male', 2: 'Female'}
merge2['Gender'] = merge2['PESEX'].map(gender_map)

race_map = {
    1: "White Only", 2: "Black Only", 3: "American Indian, Alaskan Native Only",
    4: "Asian Only", 5: "Hawaiian/Pacific Islander Only", 6: "White-Black",

}
merge2['Race'] = merge2['PTDTRACE'].map(race_map)

# Recode HRFS12MC for low food security among children
merge2['HRFS12MC_coded'] = merge2['HRFS12MC'].apply(lambda x: 0 if x == 1 else (1 if x in [2, 3] else np.nan))

# Group by Gender and Race for analysis, excluding NaN values in the process
grouped_food_security = merge2.groupby(['Gender', 'Race'], dropna=True)['HRFS12MC_coded'].mean().reset_index().sort_values(by='HRFS12MC_coded', ascending=False)

print("Proportion of Low Food Security among Children by Gender and Race (Highest to Lowest):")
print(grouped_food_security)



pivot_table = grouped_food_security.pivot_table(index="Race", columns="Gender", values="HRFS12MC_coded", aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Proportion of Food Security by Gender and Race')
plt.ylabel('Race')
plt.xlabel('Gender')
plt.show()



merge2_clean = merge2.dropna(subset=['HEFAMINC_x', 'PTDTRACE', 'PESEX', 'HRFS12MC'])

# Exclude negative values in 'HEFAMINC_x'
merge2_clean = merge2_clean[merge2_clean['HEFAMINC_x'] >= 0]

# Recode 'HRFS12MC' to binary outcome for food insecurity
merge2_clean['Food_Insecurity'] = np.where(merge2_clean['HRFS12MC'] > 1, 1, 0)

# Convert 'PESEX' to binary (0 for Male, 1 for Female)
merge2_clean['Gender'] = merge2_clean['PESEX'].apply(lambda x: 1 if x == 2 else 0)

# encode 'PTDTRACE' for race categories
race_dummies = pd.get_dummies(merge2_clean['PTDTRACE'], prefix='Race', drop_first=True)
merge2_clean = pd.concat([merge2_clean, race_dummies], axis=1)

# Prepare the independent variables (X)
X = merge2_clean[['Gender', 'HEFAMINC_x'] + list(race_dummies.columns)].copy()

# Convert boolean columns to int (1 for True, 0 for False)
for col in X.columns:
    if X[col].dtype == 'bool':
        X.loc[:, col] = X[col].astype(int)  # Use .loc[] to specify operation clearly

# Ensure the dependent variable (y) is in an appropriate numeric format
y = merge2_clean['Food_Insecurity'].astype(int)

# Add a constant to X for the intercept
X = sm.add_constant(X)

model = sm.Logit(y, X).fit(method='lbfgs', maxiter=1000)

print(model.summary())


Race Variables: The inclusion of numerous race categories (Race_2.0 to Race_26.0) allows the model to account for variations in food insecurity across different racial groups, compared to a baseline group. These coefficients show significant differences in the likelihood of experiencing food insecurity among different racial groups. For example:

Race_2.0, Race_5.0, and Race_6.0 have positive coefficients, indicating higher odds of food insecurity compared to the baseline.
Race_4.0 and Race_8.0 have negative coefficients, suggesting lower odds of food insecurity relative to the baseline.

#Interesting Findings:
The model's Log-Likelihood is considerably higher than that of the LL-Null, suggesting that the model fits the data significantly better than a model with no predictors.

HEFAMINC: The coefficient for HEFAMINC is -0.1893 with a very significant z-score of -86.042, indicating a strong negative association with food insecurity. This suggests that as Family income raises food insecrity decreases.

Gender is a statistically significant predictor of food insecurity, with the specific coded gender females as 1 being more susceptible to food insecurity than the reference gender group males.
