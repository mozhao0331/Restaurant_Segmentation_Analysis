# Sitewise Analytics Restaurant Segmentation Proposal Report Pipeline
# author: UBC MDS Cohort7 DSCI591 Sitewise Analytics Restaurant Segmentation Group
# date: 2023-05-12

# PandaDoc install: https://github.com/jgm/pandoc/releases to render the report pdf
all: doc/Final_Report.pdf
.PHONY: smoothie_king
smoothie_king: img/smoothie_king/%.png
.PHONY: subway_usa
subway_usa: img/subway_usa/%.png
.PHONY: proposal
proposal: doc/Proposal_Report.pdf

# Use Create EDA plots
# Save generated images in img folder (no dependency)
img/eda/smoothie_category_bar_plot.png img/eda/market_size_stackstacked_bar_plot.png img/eda/store_density_stackstacked_bar_plot.png img/eda/subway_us_store_density_bar_plot.png img/eda/subway_us_market_size_bar_plot.png img/eda/subway_canada_store_density_bar_plot.png img/eda/subway_canada_market_size_bar_plot.png:
	python src/helper_create_eda_figure.py

# Render proposal report
doc/Proposal_Report.pdf: doc/Proposal_Report.Rmd img/eda/smoothie_category_bar_plot.png img/eda/market_size_stackstacked_bar_plot.png img/eda/store_density_stackstacked_bar_plot.png img/eda/subway_us_store_density_bar_plot.png img/eda/subway_us_market_size_bar_plot.png img/eda/subway_canada_store_density_bar_plot.png img/eda/subway_canada_market_size_bar_plot.png img/info/smoothie_flow_chart.png img/info/subway_flow_chart.png img/info/timeline.png
	Rscript -e "rmarkdown::render('doc/Proposal_Report.Rmd')"

# Preprocess data for Smoothie King
data/Smoothie_King_Preprocessed/processed_demographic.csv data/Smoothie_King_Preprocessed/processed_poi.csv data/Smoothie_King_Preprocessed/processed_trade_area.csv:
	python src/smoothie_king_preprocess_data.py

# Fit and save the Smoothie King model
data/Smoothie_King_Preprocessed/train_df.csv data/Smoothie_King_Preprocessed/test_df.csv model_joblib/smoothie_king/rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_ovr_model.joblib model_joblib/smoothie_king/hard_voting_model.joblib: data/Smoothie_King_Preprocessed/processed_poi.csv data/Smoothie_King_Preprocessed/processed_trade_area.csv
	python src/smoothie_king_build_model.py

# SHAP interpretation Smoothie King model
img/smoothie_king/%.png: data/Smoothie_King_Preprocessed/train_df.csv data/Smoothie_King_Preprocessed/test_df.csv model_joblib/smoothie_king/rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_model.joblib model_joblib/smoothie_king/l1_reg_rf_ovr_model.joblib model_joblib/smoothie_king/hard_voting_model.joblib
	python src/smoothie_king_model_interpret.py

# Preprocess data for Subway USA
data/Subway_USA_Preprocessed/subway_usa_processed_test.csv data/Subway_USA_Preprocessed/subway_usa_processed_train.csv:
	python src/subway_usa_preprocess_data.py

# Fit and save the Subway USA model, label data and verify using map
model_joblib/subway_usa/fcm_model.joblib data/Subway_USA_Preprocessed/train_df_with_labels.csv data/Subway_USA_Preprocessed/test_df_with_labels.csv: data/Subway_USA_Preprocessed/subway_usa_processed_test.csv data/Subway_USA_Preprocessed/subway_usa_processed_train.csv
	python src/subway_usa_build_model.py

# Interpretation Subway USA model
img/subway_usa/%.png: data/Subway_USA_Preprocessed/train_df_with_labels.csv data/Subway_USA_Preprocessed/test_df_with_labels.csv
	python src/subway_usa_model_interpret.py

# Render final report
doc/Final_Report.pdf: doc/Final_Report.Rmd img/info/smoothie_king_pipeline.png img/info/subway_pipeline.png img/smoothie_king/test_cm.png img/smoothie_king/random_forest/HOME_shap_summary_plot.png img/smoothie_king/l1_reg_random_forest/HOME_shap_summary_plot.png img/subway_usa/cluster_0_summary_plot.png img/subway_usa/cluster_1_summary_plot.png img/subway_usa/cluster_2_summary_plot.png img/subway_usa/cluster_3_summary_plot.png img/subway_usa/cluster_4_summary_plot.png
	Rscript -e "rmarkdown::render('doc/Final_Report.Rmd')"

clean: 
	rm -f doc/Final_Report.pdf
	rm -rf data/Smoothie_King_Preprocessed/
	rm -rf model_joblib/smoothie_king/
	rm -rf img/smoothie_king/
	rm -rf data/Subway_USA_Preprocessed/
	rm -rf model_joblib/subway_usa/
	rm -rf img/subway_usa/
	# rm -rf img/eda/smoothie_category_bar_plot.png 
	# rm -rf img/eda/market_size_stackstacked_bar_plot.png 
	# rm -rf img/eda/store_density_stackstacked_bar_plot.png
	# rm -rf img/eda/subway_us_store_density_bar_plot.png 
	# rm -rf img/eda/subway_us_market_size_bar_plot.png 
	# rm -rf img/eda/subway_canada_store_density_bar_plot.png 
	# rm -rf img/eda/subway_canada_market_size_bar_plot.png

clean_sk:
	rm -rf data/Smoothie_King_Preprocessed/
	rm -rf model_joblib/smoothie_king/
	rm -rf img/smoothie_king/

clean_sb:
	rm -rf data/Subway_USA_Preprocessed/
	rm -rf model_joblib/subway_usa/
	rm -rf img/subway_usa/
	