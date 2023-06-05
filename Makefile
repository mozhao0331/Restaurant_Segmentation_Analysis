# Sitewise Analytics Restaurant Segmentation Proposal Report Pipeline
# author: UBC MDS Cohort7 DSCI591 Sitewise Analytics Restaurant Segmentation Group
# date: 2023-05-12

# PandaDoc install: https://github.com/jgm/pandoc/releases to render the report pdf
all: doc/Proposal_Report.pdf
.PHONY: smoothie_king
smoothie_king: img/smoothie_king/%.png

# Use Create EDA plots
# Save generated images in img folder (no dependency)
img/eda/smoothie_category_bar_plot.png img/eda/market_size_stackstacked_bar_plot.png img/eda/store_density_stackstacked_bar_plot.png img/eda/subway_us_store_density_bar_plot.png img/eda/subway_us_market_size_bar_plot.png img/eda/subway_canada_store_density_bar_plot.png img/eda/subway_canada_market_size_bar_plot.png:
	python src/helper_create_eda_figure.py

# Render final report
doc/Proposal_Report.pdf: doc/Proposal_Report.Rmd img/eda/smoothie_category_bar_plot.png img/eda/market_size_stackstacked_bar_plot.png img/eda/store_density_stackstacked_bar_plot.png img/eda/subway_us_store_density_bar_plot.png img/eda/subway_us_market_size_bar_plot.png img/eda/subway_canada_store_density_bar_plot.png img/eda/subway_canada_market_size_bar_plot.png img/info/smoothie_flow_chart.png img/info/subway_flow_chart.png img/info/timeline.png
	Rscript -e "rmarkdown::render('doc/Proposal_Report.Rmd')"

# Preprocess data for Smoothie King
data/Smoothie\ King/processed_demographic.csv data/Smoothie\ King/processed_poi.csv data/Smoothie\ King/processed_trade_area.csv:
	python src/smoothie_king_preprocess_data.py

# Fit and save the Smoothie King model
data/Smoothie\ King/train_df.csv data/Smoothie\ King/test_df.csv model_joblib/smoothie_king/rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_ovr_model.joblib model_joblib/smoothie_king/hard_voting_model.joblib: data/Smoothie\ King/processed_poi.csv data/Smoothie\ King/processed_trade_area.csv
	python src/smoothie_king_build_model.py

# SHAP interpretation Smoothie King model
img/smoothie_king/%.png: data/Smoothie\ King/train_df.csv data/Smoothie\ King/test_df.csv model_joblib/smoothie_king/rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_ovr_model.joblib model_joblib/smoothie_king/hard_voting_model.joblib
	python src/smoothie_king_model_interpret.py

clean: 
	rm -rf doc/Proposal_Report.pdf
	rm -rf img/eda/smoothie_category_bar_plot.png 
	rm -rf img/eda/market_size_stackstacked_bar_plot.png 
	rm -rf img/eda/store_density_stackstacked_bar_plot.png
	rm -rf img/eda/subway_us_store_density_bar_plot.png 
	rm -rf img/eda/subway_us_market_size_bar_plot.png 
	rm -rf img/eda/subway_canada_store_density_bar_plot.png 
	rm -rf img/eda/subway_canada_market_size_bar_plot.png

clean_sk:
	rm -f data/Smoothie\ King/processed_demographic.csv
	rm -f data/Smoothie\ King/processed_poi.csv
	rm -f data/Smoothie\ King/processed_trade_area.csv
	rm -f data/Smoothie\ King/train_df.csv
	rm -f data/Smoothie\ King/test_df.csv
	rm -rf model_joblib/smoothie_king/
	rm -rf img/smoothie_king/