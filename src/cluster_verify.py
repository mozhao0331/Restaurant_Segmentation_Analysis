
import urllib
import urllib.parse
import urllib.request

# install selenium to interact with Chrome browser ('pip install selenium')
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC


def main():
    latitude = 37.42199  # Latitude of place
    longitude = -122.08408 # Longitude of place
    zoom = 16  # Zoom level (1-21)

    map_url = 'https://maps.googleapis.com/maps/api/js?'
    center = '{},{}'.format(latitude, longitude)
    map_url += urllib.parse.urlencode({
        'center': center,
        'zoom': zoom,
        'size': '600x300',
        'maptype': 'roadmap',
        'key': 'AIzaSyDSIEnpgGi6s6cCMrLNSTCglznpu-9g7co'  # Replace with your Google Maps API key
    })

    # Download the chrome driver file from https://chromedriver.storage.googleapis.com/index.html
    # Ensure the driver version matches the chrome browser version
    driver = webdriver.Chrome('/usr/local/bin/chromedriver') # Path to chrome driver Unix Executable File
    driver.get("https://www.google.com/maps")
    driver.set_page_load_timeout(10) # Wait up to 10 seconds for pages to load 

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchboxinput")))

    # search_box = driver.find_element('//*[@id="searchboxinput"]')
    search_box = driver.find_element('//*[@id="searchboxinput"]')
    search_box.send_keys('New York, NY')
    search_box.submit()

    driver.execute_script('map.setZoom(14);')

    input("Press enter to quit")

if __name__ == "__main__":
    main()