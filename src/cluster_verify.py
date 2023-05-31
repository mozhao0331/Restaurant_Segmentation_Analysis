
import os
import urllib
import urllib.parse
import urllib.request

# install selenium to interact with Chrome browser ('pip install selenium')
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC


def main():
    cwd = os.getcwd()
    html_file = os.path.join(cwd, 'src/cluster_verify.html')
 
    # Download the chrome driver file from https://chromedriver.storage.googleapis.com/index.html
    # Ensure the driver version matches the chrome browser version
    driver = webdriver.Chrome('/usr/local/bin/chromedriver') # Path to chrome driver Unix Executable File
    driver.set_window_size(1540,1240)
    driver.get('file://' + html_file)

    driver.set_page_load_timeout(10) # Wait up to 10 seconds for pages to load 

    input("Press enter to quit")

if __name__ == "__main__":
    main()