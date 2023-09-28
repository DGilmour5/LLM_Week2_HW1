
import os
from scrapy.selector import Selector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from tqdm.auto import tqdm
from uuid import uuid4

from langchain.vectorstores import Pinecone

def scrape_web(url: str,index_name: str) -> Pinecone:
    # Setup chrome for web scraping
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)

    driver.get(url)

    sel = Selector(text = driver.page_source)
    review_counts = sel.css('.lister .header span::text').extract_first().replace(',','').split(' ')[0]
    more_review_pages = int(int(review_counts)/25)

    for i in tqdm(range(more_review_pages)):
        try:
            css_selector = 'load-more-trigger'
            driver.find_element(By.ID, css_selector).click()
        except:
            pass
    

    rating_list = []
    review_date_list = []
    review_title_list = []
    author_list = []
    review_list = []
    review_url_list = []
    error_url_list = []
    error_msg_list = []
    reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')

    for d in tqdm(reviews):
        try:
            sel2 = Selector(text = d.get_attribute('innerHTML'))
            try:
                rating = sel2.css('.rating-other-user-rating span::text').extract_first()
            except:
                rating = np.NaN
            try:
                review = sel2.css('.text.show-more__control::text').extract_first()
            except:
                review = np.NaN
            try:
                review_date = sel2.css('.review-date::text').extract_first()
            except:
                review_date = np.NaN
            try:
                author = sel2.css('.display-name-link a::text').extract_first()
            except:
                author = np.NaN
            try:
                review_title = sel2.css('a.title::text').extract_first()
            except:
                review_title = np.NaN
            try:
                review_url = sel2.css('a.title::attr(href)').extract_first()
            except:
                review_url = np.NaN
            rating_list.append(rating)
            review_date_list.append(review_date)
            review_title_list.append(review_title)
            author_list.append(author)
            review_list.append(review)
            review_url_list.append(review_url)
        except Exception as e:
            error_url_list.append(url)
            error_msg_list.append(e)
    review_df = pd.DataFrame({
        'Review_Date':review_date_list,
        'Author':author_list,
        'Rating':rating_list,
        'Review_Title':review_title_list,
        'Review':review_list,
        'Review_Url':review_url
        })
    
    data = review_df

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len 
    )



    pinecone_key = os.environ["PINECONE_API_KEY"]
    pinecone_env = os.environ["PINECONE_ENV"]

    pinecone.init(
    api_key= pinecone_key,
    environment= pinecone_env
    )

    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension= 1536 ### YOU CODE HERE - REMEMBER TO USE THE SAME DIMENSION AS THE EMBEDDING MODEL (text-embedding-ada-002)
        )

    index = pinecone.GRPCIndex(index_name)

    store = LocalFileStore("./cache/")

    core_embeddings_model = OpenAIEmbeddings()

    embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings= core_embeddings_model,
        document_embedding_cache= store,
        namespace= "dg"
    )



    BATCH_LIMIT = 100

    texts = []
    metadatas = []

    for i in tqdm(range(len(data))):

        record = data.iloc[i]

        metadata = {
            'review-url': str(record["Review_Url"]),
            'review-date' : str(record["Review_Date"]),
            'author' : str(record["Author"]),
            'rating' : str(record["Rating"]),
            'review-title' : str(record["Review_Title"]),
        }

        record_texts = text_splitter.split_text(
            text= str(record["Review"]),
            )

        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        
        if len(texts) >= BATCH_LIMIT:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embedder.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embedder.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))



    text_field = "text"

    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index= index,
        embedding= core_embeddings_model,
        text_key=text_field,
    )

    return vectorstore