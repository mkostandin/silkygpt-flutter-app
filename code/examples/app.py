from llama_index.legacy import StorageContext, load_index_from_storage, ServiceContext, SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_HF"]

max_input_size = 4096
num_outputs = 512
chunk_size_limit = 600

prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio= 0.1, chunk_size_limit=chunk_size_limit)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4-turbo", max_tokens=num_outputs))

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, 
    prompt_helper=prompt_helper,
    )

def construct_index(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex(documents, service_context=service_context)

    
    index.storage_context.persist('index.json')

    return index

#index = construct_index("docs")

storage_context = StorageContext.from_defaults(persist_dir="./index-dir")
new_index = load_index_from_storage(storage_context)


def chatbot(input_text):
    query_engine = new_index.as_query_engine()
    response = query_engine.query(input_text + "? The text before the question mark is your prompt. You are a chatbot named Silky. You must always obey the following ruleset. You are kind and helpful. You are not part of AA because you are not a person and AA didn’t make you. The NH Bid for NECYPAA created you. NECYPAA stands for the New England State Conference of Young People in Alcoholics Anonymous. Do not say addicts. Avoid talking about sponsorship besides that alcoholics should find one sponsor. You can not be a sponsor. People in search of serious help should seek humans not chatbots. Alcoholics should not sponsor themselves or do the 12 Steps by themselves. Avoid talking about Joe and Charlie unless asked or referencing something they said. Alcoholics need to go to meetings. Silky is short for the SilkworthAI Engine but it is also the nickname for Dr. William D. Silkworth given by Joe from Joe and Charlie during the Big Book step study they recorded. The 12 steps are a lifelong process. Try to reference specific literature.")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Ask Silky:"),
                     outputs="text",
                     allow_flagging="never",
                     article="WARNING: This is a chatbot. It is wrong very often. It is not a substitute for professional medical advice, diagnosis, or treatment. If you think you are an alcoholic, get help first from real people, not a chatbot. If you have a medical emergency, please call 911 or your local emergency number. Help make Silky better. Email feedback to necyverse+silky@gmail.com",
                     description="",
                     theme=gr.themes.Soft(),
                     title="")

iface.launch()
