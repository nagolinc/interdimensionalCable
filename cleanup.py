import os
import sys
import sqlite3
import shutil
import time
import datetime

#remove all files that start with the prefix  media.db
def remove_media_db():
    for filename in os.listdir("."):
        if filename.startswith("media.db"):
            os.remove(filename)

#remove all files in ./static/samples except .gitkeep
def remove_samples():
    for filename in os.listdir("./static/samples"):
        if filename != ".gitkeep":
            os.remove("./static/samples/" + filename)

remove_media_db()
remove_samples()
