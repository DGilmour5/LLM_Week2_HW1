# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
import openai  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv
import asyncio
from langchain.llms.openai import OpenAIChat
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

import time
import warnings
warnings.filterwarnings("ignore")
import wandb

from scrape_web import scrape_web


load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


#wandb.init(project="Doug's Visibility Example")




""" chat_openai = ac()
retrieval_augmented_qa_pipeline = rqa.RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai,
    wandb_project="Doug's LLM Visibility Example"
) """



llm = OpenAIChat(
    model="gpt-3.5-turbo", 
    temperature=0.5,
)

retriever = scrape_web("https://www.imdb.com/title/tt1517268/reviews/?ref_=tt_ov_rt","dg1").as_retriever()

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


    content = f"Doug is saying something: Welcome"
    await cl.Message(
        content=content,
    ).send()


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: str):
    settings = cl.user_session.get("settings")



    #response = retrieval_augmented_qa_pipeline.run_pipeline(message)
    user_input = {"query" : message}
    response = qa_with_sources_chain(user_input)
    msg = cl.Message(content=response["result"])

    # Send and close the message stream
    await msg.send()
