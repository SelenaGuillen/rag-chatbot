# rag-chatbot

```
rag-chatbot/
│
├── app/
│   ├── client.py
│   ├── components.py
│   ├── main.py
│   ├── routes.py
│   └── services.py
│
└── data/
    └── txt_docs/
```
This structure is what I'm used to with Java, with routes being controllers, services being the service layer, and client calling source API/data. I did try to keep the RAG components separate just to keep things cleaner.
To start, just ensure the CO_API_KEY is included in .env and run `uvicorn app.main:app --reload` 

## RAG 
- Used LLamaIndex over LangChain for simplicity
- Command R, passed in by default through cohere, was chosen because it is optimized for RAG and has an easy-to-use chat api. 
- Used SentenceSplitter to chunk because it has a preference for complete sentences and is less likely to output fragments.


## What did you think of the project?
Fun, but difficult for sure. This was my first time really getting into the implementation details for a RAG-based architecture, so it might still be missing a few of the key components. 
I did enjoy learning about RAG and even explored other variations such as Order-Preserving RAG (not implemented, just looked into). 

## What challenges did you encounter, and how did you address them?
There is so much  content out there. I found myself researching various methods, but ultimately decided to go with something simple due to the time constraint. I thought about using LangChain, but it seemed a bit more complex than LlamaIndex for what the project was. I also noticed that some of the functions/components were deprecated, but generative AI helped with that. 

## How would you improve your solution or approach if given more time?
I would add validation (pydantic), error handling, scalability (replace the in-memory vector store - that isn't finished), and looking into RAG best practices. 
I also need to research how to test an app like this, which would most likely include retrieval speed and efficiency.

## Any additional insights or reflections?
I'm glad I did this even though it's not as fully fleshed out as I would like it to be. I did spend most of the time on courses and reading about how it works vs. coding it, which is great for me. I think it's really valuable and I'm excited to keep learning more. 
