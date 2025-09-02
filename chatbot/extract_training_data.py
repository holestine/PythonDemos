from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders      import AsyncChromiumLoader
from pypdf                                     import PdfReader

# Suppresses warning
import os
os.environ['USER_AGENT'] = 'custom_agent'

# Used to visually identify different chunks of data
data_delimiter = "\n\n#####\n\n"

# Modify these to include sites with relevant data
URLS = [
    'https://financialcrimeacademy.org/fraud-detection-methods/',
    'https://www.fraud.com/post/5-fraud-detection-methods-for-every-organization',
    'https://www.sas.com/en_us/insights/articles/risk-fraud/strategies-fraud-detection.html',
    'https://www.tookitaki.com/compliance-hub/a-comprehensive-guide-to-financial-fraud-detection-and-prevention'
]

class DataScraper():

    def __init__(self):
        pass

    def extract_webpage_text(self, urls=URLS, tags=["h1", "h2", "h3", "p"]):
        
        # Load HTML content using AsyncChromiumLoader
        loader = AsyncChromiumLoader(urls)
        docs = loader.load()

        # Transform the loaded HTML using BeautifulSoupTransformer
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs, tags_to_extract=tags
        )

        data = [doc.page_content for doc in docs_transformed]
        
        return data

    def extract_pdf_text(self, pdf_dir='PDFs/'):

        pdfs = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith(".pdf")]

        data = []

        for pdf in pdfs:
            reader = PdfReader(pdf)

            text = []
            for page in reader.pages:
                text.append(page.extract_text())

            data.append('\n'.join(text))

        return data

def write_data(data, data_file="data/data.txt", remove_old_data=True):

    data_dir = os.path.basename(data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if remove_old_data:
        open(data_file, 'w').close() # delete file contents
    
    # Join documents with delimiter 
    data = f'{data_delimiter}'.join(x for x in data)

    # Save the data
    with open(data_file, 'a', encoding="utf-8") as file:
        file.write(data)
        file.write(f'{data_delimiter}')

if __name__ == "__main__":

    ds = DataScraper()
    web_data = ds.extract_webpage_text()
    pdf_data = ds.extract_pdf_text()
    write_data(web_data+pdf_data)
