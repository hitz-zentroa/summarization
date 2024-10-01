import json
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import re
import requests
import trafilatura

LINK_MATCHER = re.compile('==[^=]+=l')

def clean(text):
	aux = LINK_MATCHER.sub('', text).replace('=l', '').replace('=b', '').replace('=i', '')
	return ' '.join(aux.split()).strip()

def get_articles_urls(html):
	soup = BeautifulSoup(html, "html.parser")
	articles = soup.find_all("article")
	return [a.find("a").get("href") for a in articles]

def create_instance(url, category):
	response = requests.get(url)
	soup = BeautifulSoup(response.content, "html.parser")
	date =  soup.find("div", {"class": "m-date m-date--viewer"}).text.strip()
	title = soup.find("title").text.strip()
	subtitle = soup.find("h2", {"class": "c-mainarticle__subtitle"}).text.strip()
	summary = title + ". " + subtitle
	body = soup.find("div", {"class": "c-mainarticle__body"}).text.strip()
	body = clean(body)
	fetched = trafilatura.fetch_url(url)
	text = trafilatura.extract(fetched).replace("\n", "\n\n")
	instance = {"date": date, "url": url, "category": category, "title": title, "subtitle": subtitle, "summary": summary, "text": body, "text_traf": text}
	return instance

def add_instances(articles, outfile, category):
	for a in articles:
		try:
			instance = create_instance(a,category)
			with open(outfile, "a") as o:
				json.dump(instance, o)
				o.write("\n")
		except:
			pass

# Set up the headless browser
#print("Setting up the headless browser...")
options = Options()
#options.add_argument("--headless")
#driver = webdriver.Chrome(options=options)


categories = ["euskal-herria", "ekonomia", "mundua", "kultura", "kirola", "bizigiro", "komunikazioa"]


outfile = "berria_summ.jsonl"
delay = 5  # seconds


for category in categories:
	driver = webdriver.Chrome(options=options)
	url = f"https://www.berria.eus/{category}"
	print(f"Fetching url {url}")
	driver.get(url)
	time.sleep(delay)

	# Find the button element and click it once to make bottom banner appear
	print("clicking first time")
	try:
		button = driver.find_element(By.XPATH, '//button[text()="Gehiago ikusi"]')
		button.click()
		time.sleep(delay)
	except:
		pass

	# Click to exit bottom banner
	print("clicking on bottom banner")
	try:
		button = driver.find_element(By.CLASS_NAME, "c-bottom-bar__close")
		button.click()
	except:
		pass

	# Find the button element and click it repeatedly to load many stories
	for i in range(200):
		print("clicking again for the {} time".format(i+2))
		try:
			button = driver.find_element(By.XPATH, '//button[text()="Gehiago ikusi"]')
			button.click()
			time.sleep(delay)
		except:
			pass


	html = driver.page_source
	articles = get_articles_urls(html)
	print("Saving {} articles from {}".format(len(articles), url))

	add_instances(articles, outfile, category)

