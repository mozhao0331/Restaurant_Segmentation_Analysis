
import os
import urllib
import urllib.parse
import urllib.request

# install selenium to interact with Chrome browser ('pip install selenium')
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC


def cluster_verify(cluster_coords_dict):
    cwd = os.getcwd()
    html_file = os.path.join(cwd, 'cluster_verify.html')
    driver_list = []
   
    for key in cluster_coords_dict:
        coord_list = cluster_coords_dict[key]

        # Download the chrome driver file from https://chromedriver.storage.googleapis.com/index.html
        # Ensure the driver version matches the chrome browser version
        driver = webdriver.Chrome('/usr/local/bin/chromedriver') # Path to chrome driver Unix Executable File
        driver_list.append(driver)

        driver.set_window_size(1540,1240)
        driver.get('file://' + html_file)
        driver.set_page_load_timeout(10) # Wait up to 10 seconds for pages to load 
        
        # input_field = driver.find_element_by_id("coords")
        input_field = driver.find_element(By.ID, 'coords')
        input_field.send_keys(f"{coord_list}")

        button = driver.find_element(By.ID, 'btn-show-map')
        button.click()

    # for driver in driver_list:
    #     driver.quit()
    input("Press enter to quit")