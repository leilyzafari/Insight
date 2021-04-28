import sys, re
import datetime
import time
import argparse
import logging
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains


URL_PATTERN = 'http(s)?:\/\/.?(www\.)?tripadvisor\.(com|ca|de)\/Restaurant_Review.*'

class Review():
    def __init__(self, id, date, title, user, text):
        self.id = id
        self.date = date
        self.title = title
        self.user = user
        self.text = text


class TripadvisorScraper():
    def __init__(self):
        self.language = 'en'
        self.lookup = {}

        options = webdriver.ChromeOptions()
        options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        chrome_driver_binary = r"D:\myProjects\project\scrape\chromedriver.exe"
        self.driver = webdriver.Chrome(chrome_driver_binary, chrome_options=options)

        self.i18n = {
            'en': {
                'more_btn': 'More',
                'date_format': '%B %d, %Y'
            },
            'de': {
                'more_btn': 'Mehr',
                'date_format': '%d. %B %Y'
            }
        }

    def _parse_page(self):
        reviews = []
        time.sleep(2)
        review_elements = self.driver.find_elements_by_xpath('//div[@class="review-container"]')
        try:
            show_more_button = self.driver.find_element_by_xpath("//span[@class='taLnk ulBlueLinks']")
            ActionChains(self.driver).move_to_element(show_more_button).click(show_more_button).perform()
        except NoSuchElementException:
            pass
        time.sleep(2)
        for e in review_elements:
            id = e.get_attribute('id')
            date = e.find_element_by_class_name('ratingDate').get_attribute('title')
            date = datetime.datetime.strptime(date, self.i18n[self.language]['date_format'])
            title = e.find_element_by_class_name('title').find_element_by_class_name('noQuotes').text

            user = e.find_element_by_class_name('memberOverlayLink').get_attribute('id')
            user = user[4:user.index('-')]

            text = e.find_element_by_class_name('partial_entry').text.replace('\n', '')
            reviews.append(Review(id, date, title, user, text))

        if self.driver.find_elements_by_xpath('//a[@class="nav next ui_button primary"]'):
            next_page = self.driver.find_element_by_xpath('//a[@class="nav next ui_button primary"]')
            ActionChains(self.driver).move_to_element(next_page).click(next_page).perform()

            reviews += self._parse_page()
        return reviews

    def _set_language(self, url=''):
        if 'tripadvisor.de' in url:
            self.language = 'de'
        elif 'tripadvisor.com' in url:
            self.language = 'en'
        else:
            logging.warning('Tripadvisor domain location not supported. Defaulting to English (.com)')

    def fetch_reviews(self, url, max_reviews=None, as_dataframe=True):
        reviews = []

        if not max_reviews: max_reviews = sys.maxsize
        self._set_language(url)

        if not is_valid_url(url): return logging.warning('Tripadvisor URL not valid.')
        self.driver.get(url)
        reviews += self._parse_page()
        reviews = reviews[:max_reviews]

        if as_dataframe: return pd.DataFrame.from_records([r.__dict__ for r in reviews]).set_index('id', drop=True)
        return reviews

    def close(self):
        self.driver.quit()


def is_valid_url(url):
    return re.compile(URL_PATTERN).match(url)


def get_language_by_url(url):
    if 'tripadvisor.de' in url:
        return 'de'
    elif 'tripadvisor.com' in url:
        return 'en'
    return None


def get_id_by_url(url):
    if not is_valid_url(url): return None
    match = re.compile('.*Restaurant_Review-g\d+-(d\d+).*').match(url)
    if match is None: return None
    return match.group(1)


if __name__ == '__main__':
    sys.argv = ['']
    parser = argparse.ArgumentParser(description='Scrape restaurant reviews from Tripadvisor (.com or .de).')
    parser.add_argument('-url', '--url',
                        default="https://www.tripadvisor.com/Restaurant_Review-g155019-d1308932-Reviews-Blu_Ristorante-Toronto_Ontario.html",
                        help='URL to a Tripadvisor restaurant page')
    parser.add_argument('-o', '--out', dest='outfile', help='Path for output CSV file', default='reviews.csv')
    parser.add_argument('-n', dest='max', help='Maximum number of reviews to fetch', default=sys.maxsize, type=int)
    parser.add_argument('-e', '--engine', dest='engine', help='Driver to use',
                        choices=['phantomjs', 'chrome', 'firefox'], default='chrome')
    args = parser.parse_args()
    scraper = TripadvisorScraper()
    df = scraper.fetch_reviews(args.url, args.max)
    print('Successfully fetched {} reviews.'.format(len(df.index)))
    df.to_csv(args.outfile)
