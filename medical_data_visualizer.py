import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
DATA_FILENAME = "medical_examination.csv"
df = pd.read_csv(f"{DATA_FILENAME}", sep=",", skipinitialspace=True)


# 2
df['BMI'] = df["weight"] / ((df["height"]/100)**2)
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)
df = df.drop('BMI', axis=1)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x <= 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x <= 1 else 1)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=["id"], value_vars=sorted(["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]))

    # 6
    df_cat = pd.melt(df, id_vars=["id", "cardio"], value_vars=sorted(["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]))

    # 7
    draw_cat_plot_chart = sns.catplot(x="variable", col="cardio", hue="value", data=df_cat, kind="count").set_axis_labels("variable", "total")

    # 8
    fig = draw_cat_plot_chart.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()
    df_heat = df_heat[(df_heat['ap_lo'] <= df['ap_hi'])]
    df_heat = df_heat[(df_heat['height'] >= df['height'].quantile(0.025))]
    df_heat = df_heat[(df_heat['height'] <= df['height'].quantile(0.975))]
    df_heat = df_heat[(df_heat['weight'] >= df['weight'].quantile(0.025))]
    df_heat = df_heat[(df_heat['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(11, 9))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True, linewidths=1, cmap="twilight", center=0, cbar_kws={'shrink': 0.5})


    # 16
    fig.savefig('heatmap.png')
    return fig
