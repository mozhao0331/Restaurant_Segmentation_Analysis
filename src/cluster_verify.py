
import urllib
import urllib.parse
import urllib.request

# install selenium to interact with Chrome browser ('pip install selenium')
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC


def main():
    latitude = 30.519353  # Latitude of place
    longitude = -97.650392 # Longitude of place
    zoom = 12  # Zoom level (1-21)
    place_url = f'https://www.google.com/maps/place/{latitude},{longitude}/data=!3m1!1e3!5m2!1e1!1e4'


    # map_url = 'https://maps.googleapis.com/maps/api/js?'
    # center = '{},{}'.format(latitude, longitude)
    # map_url += urllib.parse.urlencode({
    #     'center': center,
    #     'zoom': zoom,
    #     'size': '600x300',
    #     'maptype': 'roadmap',
    #     'key': 'AIzaSyDSIEnpgGi6s6cCMrLNSTCglznpu-9g7co'  # Replace with your Google Maps API key
    # })

    # Download the chrome driver file from https://chromedriver.storage.googleapis.com/index.html
    # Ensure the driver version matches the chrome browser version
    driver = webdriver.Chrome('/usr/local/bin/chromedriver') # Path to chrome driver Unix Executable File
    driver.set_window_size(1540,1240)
    driver.get(place_url)

    # Load Google Maps API to define the map object
    # driver.execute_script("""
    # var script = document.createElement('script');  
    # script.src = 'https://maps.googleapis.com/maps/api/js?key=AIzaSyDhdS3PxDvGx0KXJHVLXsQF5R7LEpCW0K4'; 
    # document.head.appendChild(script);  
    # script.onload = function() {
    #     map.setMapTypeId(google.maps.MapTypeId.SATELLITE); 
    #     map.setZoom(2);
    # };
    # """)        

    # driver.execute_async_script("""
    # window.map = null;
    # var script = document.createElement('script');
    # script.src = 'https://maps.googleapis.com/maps/api/js?key=AIzaSyDSIEnpgGi6s6cCMrLNSTCglznpu-9g7co'; 
    # document.head.appendChild(script);
    # script.onload = function() { 
    #     window.map = google.maps.Map.fromDiv(this);  
    #     window.map.setMapTypeId(google.maps.MapTypeId.SATELLITE);
    #     window.map.setZoom(2);
    # };
    # """)              
    driver.set_page_load_timeout(10) # Wait up to 10 seconds for pages to load 

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "searchboxinput")))

    # search_box = driver.find_element('//*[@id="searchboxinput"]')
    # search_box = driver.find_element(By.ID, "searchboxinput")
    # search_box.send_keys('New York, NY')
    # search_box.submit()

    input("Press enter to quit")

if __name__ == "__main__":
    main()