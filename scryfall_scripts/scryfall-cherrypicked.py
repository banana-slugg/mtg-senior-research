from enum import unique
from aiohttp import request
import scrython
import time
import requests
import os


def download_cherrypicked_art(color):
    page_count = 1
    all_data = []
    print("Grabbing {} cards".format(color))
    while True:
        time.sleep(0.2)
        page = scrython.Search(
            q="is:firstprint f:vintage -is:doublesided is:hires (is:modern OR is:new) color={} t:elemental".format(
                color
            ),
            page=page_count,
            unique="art",
        )
        all_data = all_data + page.data()
        page_count += 1
        if not page.has_more():
            break
    # return all_data
    i = 0
    path = "./card_art_cherry/{}".format(color)
    dirExist = os.path.exists(path)
    if not dirExist:
        os.makedirs(path)
    print("Downloading {} cards".format(color))
    for card in all_data:
        images = card["image_uris"]
        image_data = requests.get(images["art_crop"]).content
        with open("{}/{}.jpg".format(path, i), "wb") as handler:
            handler.write(image_data)
        i += 1


def main():
    colors = ["red", "blue", "green"]
    for color in colors:
        download_cherrypicked_art(color)


if __name__ == "__main__":
    main()
