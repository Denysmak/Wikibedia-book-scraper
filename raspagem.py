import pandas as pd
import requests
import re
import json
import tensorflow_hub as hub 
import nltk
import numpy



from bs4 import BeautifulSoup, Tag, NavigableString
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


#cria um pandas dataframe usando o arquivo csv que nós já temos
allIds = pd.read_csv('query.csv')

#transforma esse  dataframe em uma array porque é mais rápido pra acessar os valores
allIdsNp = allIds.to_numpy()


#essa função vai retirar de cada elemento da lista apenas a parte que nós precisamos, o id.
def preProcessData(data : list, numberOfElements):
    #pega os dados do sparQL, e converte para uma lista de ids
    ids = []
    for idx, ele in enumerate(data):
        if(idx >= numberOfElements):
            return ids
        #o ele vai retornar uma lista com 1 elemento, a url
        url = ele[0]
        urlArr = url.split("/")
        wikidataId = urlArr[-1]
        ids.append(wikidataId)
    return ids

#vamos usar essa função para pegar 30 ids
ids = preProcessData(allIdsNp, 30) 

def removeStopWords(string) -> str:
    word_tokens = word_tokenize(string)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ''.join(filtered_sentence)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

#get soup está funcionando
def getSoup(url):
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup








def getTitleAndSummaryFromWikipediaPage(soup):
    title = soup.select("#firstHeading")[0].text
    headers = soup.find_all(['h1', 'h2', 'h3'])
    relevantHeader = None
    for i in range(len(headers)):
        firstString = re.split('[^a-zA-Z]', headers[i].getText())[0]
        if firstString.lower() == "plot" or firstString.lower() == "summary":
            relevantHeader = headers[i]
            break 
    if relevantHeader is None:
        return None
    summary = ""
    for elem in relevantHeader.next_siblings:
        if elem.name and elem.name.startswith('h'):
            break
        if elem.name == 'p':
            summary += elem.get_text() + " "
    return title, summary

def get_wikipedia_url_from_wikidata_id(wikidata_id, lang='en', debug=False):
    import requests
    from requests import utils

    url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities'
        '&props=sitelinks/urls'
        f'&ids={wikidata_id}'
        '&format=json')
    json_response = requests.get(url).json()
    if debug: print(wikidata_id, url, json_response) 

    entities = json_response.get('entities')    
    if entities:
        entity = entities.get(wikidata_id)
        if entity:
            sitelinks = entity.get('sitelinks')
            if sitelinks:
                if lang:
                    sitelink = sitelinks.get(f'{lang}wiki')
                    if sitelink:
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            return requests.utils.unquote(wiki_url)
                else:
                    wiki_urls = {}
                    for key, sitelink in sitelinks.items():
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            wiki_urls[key] = requests.utils.unquote(wiki_url)
                    return wiki_urls
    return None


def scrape(listOfIdsToScrape, numberOfElementsToScrape, filename):
    titleAndEncodedSummaries = dict()
    numberOfSummariesAdded = 0

    for idx, idToScrape in enumerate(listOfIdsToScrape):
        if (numberOfSummariesAdded >= numberOfElementsToScrape):
            break
        try: 
            #wikipediaUrl está funcionando
            wikipediaUrl = get_wikipedia_url_from_wikidata_id(idToScrape)
            soup = getSoup(wikipediaUrl)
            res = getTitleAndSummaryFromWikipediaPage(soup)
            if (res is None):
                continue
            title, summary = res
            numberOfSummariesAdded += 1
            nonStopWordSummary = removeStopWords(summary)
            embedding = embed([nonStopWordSummary]).numpy().tolist() 
            titleAndEncodedSummaries[title] = embedding
        except:
            pass
    #end by writing the data to file
    with open(filename, 'w') as f:
        json.dump(titleAndEncodedSummaries, f)
    f.close()
    print(f'Added {numberOfSummariesAdded} books')
scrape(ids, 30, 'bookTitleAndSummaries.json')


#takes in a string, removes stop words from it, and returns the string without stopwords









