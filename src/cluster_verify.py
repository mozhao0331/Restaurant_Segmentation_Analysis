
import urllib
import urllib.parse
import urllib.request

# install selenium to interact with Chrome browser ('pip install selenium')
from selenium import webdriver


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

    search_box = driver.find_element_by_xpath('//*[@id="searchboxinput"]')
    search_box.send_keys('New York, NY')
    search_box.submit()

    driver.execute_script('map.setZoom(14);')

if __name__ == "__main__":
    main()