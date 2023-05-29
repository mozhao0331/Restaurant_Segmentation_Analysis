# Sitewise Analytics Restaurant Segmentation Proposal Report Pipeline
# author: UBC MDS Cohort7 DSCI591 Sitewise Analytics Restaurant Segmentation Group
# date: 2023-05-12

all: doc/Proposal_Report.pdf
.PHONY: smoothie_king
smoothie_king: img/smoothie_king/%.png

# Use Create EDA plots
# Save generated images in img folder (no dependency)
img/smoothie_category_bar_plot.png img/market_size_stackstacked_bar_plot.png img/store_density_stackstacked_bar_plot.png img/subway_us_store_density_bar_plot.png img/subway_us_market_size_bar_plot.png img/subway_canada_store_density_bar_plot.png img/subway_canada_market_size_bar_plot.png:
	python src/create_eda_figure.py

# Render final report
doc/Proposal_Report.pdf: doc/Proposal_Report.Rmd img/smoothie_category_bar_plot.png img/market_size_stackstacked_bar_plot.png img/store_density_stackstacked_bar_plot.png img/subway_us_store_density_bar_plot.png img/subway_us_market_size_bar_plot.png img/subway_canada_store_density_bar_plot.png img/subway_canada_market_size_bar_plot.png img/smoothie_flow_chart.png img/subway_flow_chart.png img/timeline.png
	Rscript -e "rmarkdown::render('doc/Proposal_Report.Rmd')"

# Preprocess data for Smoothie King
data/Smoothie\ King/processed_demographic.csv data/Smoothie\ King/processed_poi.csv data/Smoothie\ King/processed_trade_area.csv:
	python src/preprocess_data.py

# Fit and save the Smoothie King model
data/Smoothie\ King/train_df.csv data/Smoothie\ King/test_df.csv model_joblib/rf_model.joblib model_joblib/l1_reg_rf_model.joblib model_joblib/l1_reg_rf_ovr_model.joblib model_joblib/hard_voting_model.joblib: data/Smoothie\ King/processed_poi.csv data/Smoothie\ King/processed_trade_area.csv
	python src/build_smoothie_king_model.py

# SHAP interpretation Smoothie King model
img/smoothie_king/%.png: data/Smoothie\ King/train_df.csv data/Smoothie\ King/test_df.csv model_joblib/rf_model.joblib model_joblib/l1_reg_rf_model.joblib model_joblib/l1_reg_rf_ovr_model.joblib model_joblib/hard_voting_model.joblib
	python src/interpret_smoothie_king_model.py

clean: 
	rm -rf doc/Proposal_Report.pdf
	rm -rf img/smoothie_category_bar_plot.png 
	rm -rf img/market_size_stackstacked_bar_plot.png 
	rm -rf img/store_density_stackstacked_bar_plot.png
	rm -rf img/subway_us_store_density_bar_plot.png 
	rm -rf img/subway_us_market_size_bar_plot.png 
	rm -rf img/subway_canada_store_density_bar_plot.png 
	rm -rf img/subway_canada_market_size_bar_plot.png

clean_sk:
	rm -rf model_joblib/
	rm -rf img/smoothie_king/