import urllib.request
import os

url = "https://raw.githubusercontent.com/pjreddie/mnist-csv-png/master/test/4/00004.png"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response, open('mnist_4.png', 'wb') as out_file:
        out_file.write(response.read())
    print("Downloaded mnist_4.png")
except Exception as e:
    print(f"Failed to download: {e}")
    # Fallback to another URL
    url2 = "https://raw.githubusercontent.com/teavanist/MNIST-JPG/master/data/test/4/104.jpg"
    req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req2) as response, open('mnist_4.jpg', 'wb') as out_file:
            out_file.write(response.read())
        print("Downloaded mnist_4.jpg")
    except Exception as e2:
        print(f"Failed fallback: {e2}")
