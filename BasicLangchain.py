
#creating the text file with the input string
"""
Input='standard, xxhash, uuid-utils, truststore, tenacity, sqlalchemy, sniffio, python-dotenv, propcache, ormsgpack, orjson, multidict, langchain-protocol, jsonpointer, httpx-sse, frozenlist, distro, aiohappyeyeballs, yarl, requests-toolbelt, jsonpatch, httpcore2, aiosignal, pydantic-settings, httpx2, aiohttp, langsmith, langchain-core, langgraph-sdk, langgraph-checkpoint, langchain-text-splitters, langgraph-prebuilt, langchain-classic, langgraph, langchain-community, langchainSuccessfully installed aiohappyeyeballs-2.7.1 aiohttp-3.14.4 aiosignal-1.4.0 distro-1.9.0 frozenlist-1.8.0 httpcore2-2.13.1 httpx-sse-0.4.3 httpx2-2.13.1 jsonpatch-1.33 jsonpointer-3.1.1 langchain-1.4.3 langchain-classic-1.0.8 langchain-community-0.4.2 langchain-core-1.6.6 langchain-protocol-0.0.19 langchain-text-splitters-1.1.3 langgraph-1.2.13 langgraph-checkpoint-4.2.0 langgraph-prebuilt-1.1.0 langgraph-sdk-0.4.5 langsmith-0.14.4 multidict-6.9.1 orjson-3.12.0 ormsgpack-1.12.2 propcache-0.5.4 pydantic-settings-2.15.0 python-dotenv-1.2.4 requests-toolbelt-1.0.0 sniffio-1.3.1 sqlalchemy-2.1.3 tenacity-9.1.4 truststore-0.10.4 uuid-utils-0.17.1 xxhash-4.0.1 yarl-1.25.1 zstandard-0.25.0'

with open('install.txt', 'w') as f:
    f.write(Input)
"""
# load the text file and print the number of pages and the content of the first page
"""
from langchain_community.document_loaders import TextLoader

loader = TextLoader("install.txt")
pages = loader.load()
print("Number of pages:", len(pages))
print("Content:")
print(pages[0].page_content)
"""
'''
from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.langchain.com/oss/python/langchain/overview"

loader = WebBaseLoader(url)

documents = loader.load()

for document in documents:
    print(document.page_content)
    '''
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
text="Agent = Model + Harness. LangChain provides create_agent: a minimal, highly configurable harness. The harness is everything around the model loop: the prompt, the tools, and any middleware that shapes behavior. Start with the primitives and compose exactly what your use case needs. Supports OpenAI, Anthropic, Google, and more."
chunks = splitter.split_text(text)
print("Number of pages:", len(chunks))
print("Content:")
for i, chunk in enumerate(chunks):
    print(f"Page {i+1}:")
    print(chunk)