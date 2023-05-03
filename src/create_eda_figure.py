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

if __name__ == "__main__":
    main()