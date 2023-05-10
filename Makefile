# Sitewise Analytics Restaurant Segmentation Proposal Report Pipeline
# author: UBC MDS Cohort7 DSCI591 Sitewise Analytics Restaurant Segmentation Group
# date: 2023-05-12

all: doc/Proposal_Report.pdf

# Use Create EDA plots
# Save generated images in img folder (no dependency)
img/smoothie_category_bar_plot.png img/market_size_stackstacked_bar_plot.png img/store_density_stackstacked_bar_plot.png img/subway_us_store_density_bar_plot.png img/subway_us_market_size_bar_plot.png img/subway_canada_store_density_bar_plot.png img/subway_canada_market_size_bar_plot.png:
	python src/create_eda_figure.py

# Render final report
doc/Proposal_Report.pdf: doc/Proposal_Report.Rmd img/smoothie_category_bar_plot.png img/market_size_stackstacked_bar_plot.png img/store_density_stackstacked_bar_plot.png img/subway_us_store_density_bar_plot.png img/subway_us_market_size_bar_plot.png img/subway_canada_store_density_bar_plot.png img/subway_canada_market_size_bar_plot.png img/smoothie_flow_chart.png img/subway_flow_chart.png img/timeline.png
	Rscript -e "rmarkdown::render('doc/Proposal_Report.Rmd')"

clean: 
	rm -rf doc/Proposal_Report.pdf
	rm -rf img/smoothie_category_bar_plot.png 
	rm -rf img/market_size_stackstacked_bar_plot.png 
	rm -rf img/store_density_stackstacked_bar_plot.png
	rm -rf img/subway_us_store_density_bar_plot.png 
	rm -rf img/subway_us_market_size_bar_plot.png 
	rm -rf img/subway_canada_store_density_bar_plot.png 
	rm -rf img/subway_canada_market_size_bar_plot.png