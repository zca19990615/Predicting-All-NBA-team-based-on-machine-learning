# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:55:38 2024

@author: 61967
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

# User-agent string to mimic a web browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

savepath = 'C:/Users/61967/Desktop/机器学习2'
filename = 'prosportstransactions_scrape_missedgames_2004_2023.csv'

url = "https://www.prosportstransactions.com/basketball/Search/SearchResults.php?Player=&Team=&BeginDate=2004-10-01&EndDate=2023-07-01&InjuriesChkBx=yes&Submit=Search"

response = requests.get(url, headers=headers)  # Add headers to your request
print(response)  # Response [200] means it went through

# Continue with your scraping as before if the response is successful...
if response.status_code == 200:
    # Your scraping code here
    pass
else:
    print(f"Failed to retrieve the webpage: {response.status_code}")


html_content = response.text
print(html_content)  # 慎重，因为这可能是一个非常大的输出
