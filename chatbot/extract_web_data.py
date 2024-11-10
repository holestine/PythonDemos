from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncChromiumLoader

# Supresses warning
import os
os.environ['USER_AGENT'] = 'myagent'

# Modify these to include sites with relevant data
URLS = [
    'https://financialcrimeacademy.org/fraud-detection-methods/',
    'https://www.fraud.com/post/5-fraud-detection-methods-for-every-organization',
    'https://www.sas.com/en_us/insights/articles/risk-fraud/strategies-fraud-detection.html',
    'https://www.tookitaki.com/compliance-hub/a-comprehensive-guide-to-financial-fraud-detection-and-prevention'
]

def extract_webpage_data(urls=URLS, out_file="data.txt", tags=["h1", "h2", "h3", "p"]):
    
    # Load HTML content using AsyncChromiumLoader
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()

    # Transform the loaded HTML using BeautifulSoupTransformer
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=tags
    )

    data = [doc.page_content for doc in docs_transformed]
    data = ''.join(str(x+'\n\n') for x in data)
    with open(out_file, 'w', encoding="utf-8") as file:
        file.write(data)

if __name__ == "__main__":
    extract_webpage_data()
