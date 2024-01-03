from enum import unique
from aiohttp import request
import scrython
import time
import requests
import os


def download_all_art(color):
    page_count = 1
    all_data = []
    print("Grabbing {} cards".format(color))
    while True:
        time.sleep(0.2)
        page = scrython.Search(
            q="is:firstprint f:modern -is:doublesided is:hires (is:modern OR is:new) color={}".format(
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
    path = "./card_art/{}".format(color)
    dirExist = os.path.exists(path)
    if not dirExist:
        os.makedirs(path)
    print("Downloading {} cards".format(color))
    for card in all_data:
        images = card["image_uris"]
        image_data = requests.get(images["art_crop"]).content
        with open("{}/{}{}.jpg".format(path, color, i), "wb") as handler:
            handler.write(image_data)
        i += 1


def main():
    colors = ["white", "blue", "black", "red", "green"]
    for color in colors:
        download_all_art(color)


if __name__ == "__main__":
    main()
