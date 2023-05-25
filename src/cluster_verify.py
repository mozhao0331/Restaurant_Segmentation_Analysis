
import urllib
import urllib.parse
import urllib.request


def main():
    latitude = 37.42199  # Latitude of place
    longitude = -122.08408 # Longitude of place
    zoom = 16  # Zoom level (1-21)

    map_url = 'https://maps.googleapis.com/maps/api/staticmap?'
    center = '{},{}'.format(latitude, longitude)
    map_url += urllib.parse.urlencode({
        'center': center,
        'zoom': zoom,
        'size': '600x300',
        'maptype': 'roadmap',
        'key': 'YOUR_API_KEY'  # Replace with your Google Maps API key
    })

    map_file = urllib.request.urlopen(map_url)
    map_img = map_file.read()

    with open('map.png', 'wb') as f:
        f.write(map_img)

if __name__ == "__main__":
    main()