import os
import pandas as pd
import matplotlib.pyplot as plt

DIR = "data/"

def merge_df(restaurant):
    demographic = pd.read_csv(DIR + restaurant + "demographic_variables.csv")
    poi = pd.read_csv(DIR + restaurant + "poi_variables.csv")
    try:
        stores = pd.read_csv(DIR + restaurant + "stores.csv")
    except:
        stores = pd.read_csv(DIR + restaurant + "stores.csv", encoding="ISO-8859-1")
    merged = stores.merge(demographic).merge(poi)
    return merged

def create_bar_chart(df, col_name, xlab, title, out_dir):
    counts = df[col_name].value_counts()
    plt.figure(figsize=(9,7))
    fig = plt.bar(counts.index, counts.values)
    plt.xlabel(xlab)
    plt.ylabel("Number of Stores")
    plt.title(title)
    plt.xticks(rotation=30)
    plt.bar_label(fig, label_type="edge")
    try:
        plt.savefig("img/" + out_dir + "_bar_plot.png", bbox_inches="tight")
    except:
        os.mkdir("img/")
        plt.savefig("img/" + out_dir + "_bar_plot.png", bbox_inches="tight")

def create_stack_bar_chart(df, col_name, title, out_dir):
    plt.figure(figsize=(9,7))
    df = df[[col_name, 'category']]
    grouped = df.groupby(['category', col_name]).value_counts()
    pivoted = grouped.unstack(col_name).fillna(0)
    totals = pivoted.sum(axis=1)
    pivoted = pivoted.div(totals, axis=0) * 100

    fig, ax = plt.subplots()
    pivoted.plot(kind='bar', stacked=True, ax=ax)

    # add text
    # for container in ax.containers:
    #     ax.bar_label(container, labels=[f'{x:.1f}%' for x in container.datavalues])

    ax.set_xlabel('Category')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    try:
        plt.savefig("img/" + out_dir + "stacked_bar_plot.png", bbox_inches="tight")
    except:
        os.mkdir("img/")
        plt.savefig("img/" + out_dir + "stacked_bar_plot.png", bbox_inches="tight")

def create_boxplot(df, col_name, ylabel, title, out_dir):
    plt.figure(figsize=(9,7))
    fig = df.boxplot(
        column=[col_name], 
        by="category",
        xlabel="Category",
        ylabel=ylabel
    ).get_figure()
    plt.title(title)
    plt.suptitle("")
    try:
        fig.savefig("img/" + out_dir + "_boxplot.png", bbox_inches="tight")
    except:
        os.mkdir("img/")
        fig.savefig("img/" + out_dir + "_boxplot.png", bbox_inches="tight")

def create_scatterplot(df, x_col, y_col, groupby_col, xlab, ylab, title, out_dir):
    plt.figure(figsize=(10, 10))
    categories = df[groupby_col].unique().tolist()
    fig, ax = plt.subplots(4, 2,sharex=True, sharey=True, figsize=(10, 14))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    row = col = 0
    for category in categories:
        to_plot = df.query("store_density == @category")
        ax[row, col].scatter(to_plot[x_col], to_plot[y_col], alpha=0.4)
        ax[row, col].set_ybound(0, 200000000)
        ax[row, col].set_xbound(0, 1)
        ax[row, col].set_title(category)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.suptitle(title)
        col += 1
        if col == 2:
            row += 1
            col = 0
    try:
        fig.savefig("img/" + out_dir + "_scatterplot.png", bbox_inches="tight")
    except:
        os.mkdir("img/")
        fig.savefig("img/" + out_dir + "_scatterplot.png", bbox_inches="tight")

def main():
    SMOOTHIE = "Smoothie King/smoothie_king_"
    US_SUBWAY = "Subway USA/subway_usa_"
    CAN_SUBWAY = "Subway CAN/subway_can_"
    smoothie_df = merge_df(SMOOTHIE)
    subway_us_df = merge_df(US_SUBWAY)
    subway_can_df = merge_df(CAN_SUBWAY)
    create_bar_chart(
        smoothie_df, "category", "Category", "Smoothie King Category Bar Chart", "smoothie_category"
    )
    create_bar_chart(
        smoothie_df, "store_density", "Store Density", "Smoothie King Store Density Bar Chart", "smoothie_store_density"
    )
    create_bar_chart(
        smoothie_df, "market_size", "Market Size", "Smoothie King Market Size Bar Chart", "smoothie_market_size"
    )
    create_bar_chart(
        subway_us_df, "market_size", "Market Size", "Subway USA Market Size Bar Chart", "subway_us_market_size"
    )
    create_bar_chart(
        subway_us_df, "store_density", "Store Density", "Subway USA Store Density Bar Chart", "subway_us_store_density"
    )
    create_bar_chart(
        subway_can_df, "market_size", "Market Size", "Subway Canada Market Size Bar Chart", "subway_canada_market_size"
    )
    create_bar_chart(
        subway_can_df, "store_density", "Store Density", "Subway Canada Store Density Bar Chart", "subway_canada_store_density"
    )
    create_stack_bar_chart(
        smoothie_df, "market_size", "Stacked Bar Chart by Market Size and Category", "market_size_stack"
    )
    create_stack_bar_chart(
        smoothie_df, "store_density", "Stacked Bar Chart by Store Density and Category", "store_density_stack"
    )
    create_boxplot(
        smoothie_df, "centerxy_gla_1mi", "Gross leasable area within 1 mile radius", 
        "Boxplot of Gross Leasable Area within 1 mile by Category", "centerxy_gla_1mi_boxplot"
    )
    create_boxplot(
        smoothie_df, "gdp_10mi", "GDP within 10 mile radius", "Boxplot of GDP within 10 mile radius by Category",
        "gdp_10mi_boxplot"
    )
    create_scatterplot(
        subway_us_df, "genz_p_1mi", "spend_dinner_1mi", "store_density", "Gen Z Population Percentage in 1 mile radius",
        "Amount spent on dinner in 1 mile radius", "Scatter plot of Percent Gen Z Population vs Amount Spent on Dinner",
        "genz_p_vs_dinner"
    )

if __name__ == "__main__":
    main()