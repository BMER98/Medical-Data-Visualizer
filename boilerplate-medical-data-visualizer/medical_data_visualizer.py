import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv
df = pd.read_csv('medical_examination.csv')

# 2. Create the overweight column in the df variable
# Overweight is calculated as weight (kg) / height (m)^2 > 25
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)

# 3. Normalize data by making 0 always good and 1 always bad
# For cholesterol and gluc, set values of 1 to 0, and >1 to 1
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Draw the Categorical Plot in the draw_cat_plot function
def draw_cat_plot():
    # 5. Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat data to split it by cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Draw the categorical plot with sns.catplot()
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar").fig

    # 8. Save the figure as catplot.png
    fig.savefig('catplot.png')
    return fig

# 10. Draw the Heat Map in the draw_heat_map function
def draw_heat_map():
    # 11. Clean the data in df_heat
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15. Plot the heatmap
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, square=True, linewidths=.5, cmap="coolwarm", cbar_kws={"shrink": .5}, ax=ax)

    # 16. Save the figure as heatmap.png
    fig.savefig('heatmap.png')
    return fig
