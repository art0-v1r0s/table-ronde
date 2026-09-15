import socket

old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import logging
import os
import time

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

t0 = time.time()
key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=key)
print("Starting stream...")
for chunk in llm.stream([HumanMessage(content="Hello")]):
    pass
print(f"Finished in {time.time() - t0:.2f} seconds")
