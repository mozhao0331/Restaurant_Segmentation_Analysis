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
from selenium.webdriver.chrome.options import Options
import pyautogui

# html_path = 'src/subway_usa_cluster_verify.html'
html_path = 'subway_usa_cluster_verify.html'
chrome_driver_path = '/usr/local/bin/chromedriver'
page_title_id = "cluster_id"
input_id = "coords"
button_id = 'btn-show-map'

CLUSTER = 'Cluster'

def cluster_verify(cluster_coords_dict):
    """Function to use Selenium to open Chrome windows to verify clustering result
    Parameters
    ----------
    cluster_coords_dict : dict
        The dictionary with stores' locations and stores' ID

    Returns
    -------
    None
    """
    cwd = os.getcwd()
    html_file = os.path.join(cwd, html_path)
    driver_list = []
   
    for key in cluster_coords_dict:
        coord_list = cluster_coords_dict[key]

        # Get display resolution using PyAutoGUI
        width, height = pyautogui.size()

        # Calculate 1/3 width
        one_third_width = width // 3

        # Set Chrome options
        chrome_options = Options()  
        chrome_options.add_argument('--window-size=%d,%d' %(one_third_width, height))

        # Download the chrome driver file from https://chromedriver.storage.googleapis.com/index.html
        # Ensure the driver version matches the chrome browser version
        driver = webdriver.Chrome(chrome_driver_path, options=chrome_options) # Path to chrome driver Unix Executable File
        driver_list.append(driver)

        # driver.set_window_size(1540,1240)
        driver.get('file://' + html_file)

        wait = WebDriverWait(driver, 100) 
        page_title = wait.until(EC.presence_of_element_located((By.ID, page_title_id))) 
        input_field = wait.until(EC.presence_of_element_located((By.ID, input_id))) 
        
        page_title = driver.find_element(By.ID, page_title_id)
        page_title_txt = f'{CLUSTER} {key}'
        driver.execute_script("arguments[0].innerHTML = arguments[1];", page_title, page_title_txt)

        input_field = driver.find_element(By.ID, input_id)
        input_field.send_keys(f"{coord_list}")

        button = driver.find_element(By.ID, button_id)
        button.click()

    # for driver in driver_list:
    #     driver.quit()
    input("Press enter to quit")