"""
Script to launch a window from Chrome in Fullscreen for sandbox. Opens defined localhost URL defined in bat file
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from time import sleep

sleep(5)
options = webdriver.ChromeOptions()

options.add_experimental_option("excludeSwitches", ['enable-automation']);
service = ChromeService(executable_path=r"D:\sandbox\chromedriver\chromedriver-win64\chromedriver.exe")

output = webdriver.Chrome(service=service, options=options)

output.get("http://localhost:1234/Topography")
#output.set_window_position(2000, 0)
output.fullscreen_window()

while True:
    a = 1

