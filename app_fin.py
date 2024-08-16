from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("D:/projects/devposts/standard-chartered-plc-full-year-2023-report.pdf")
data = loader.load()
print(data[15])