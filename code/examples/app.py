from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_HF"]

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text + 'Respond helpfully. Don\'t say addicts. Alcoholics will always need to go to meetings. Alcoholics are never finished with the 12 Steps. Try to reference specific literature. ', response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Ask Silky:"),
                     outputs="text",
                     allow_flagging="never",
                     article="WARNING: This is a chatbot. It is not a person. It is a novelty designed to be fun and to help people afflicted with alcoholism and drug addiction. You can chat with me about your feelings, challenges, and goals. I can also suggest some resources and tips that may help you. Besides that, I can also entertain you with some fun and creative content, such as poems, stories, jokes, and more. However, it is not a substitute for professional medical advice, diagnosis, or treatment. If you have a medical emergency, please call 911 or your local emergency number. This chatbot is not a human and cannot understand the full context and emotions of your situation. It may make mistakes or give inappropriate responses. Please do not share any personal information with this chatbot. This chatbot may provide links to external resources that may be helpful for you, but it is not responsible for the content, quality of these resources, or of the quality and content it produces. Please use them at your own discretion. By using this chatbot, you agree to these terms and conditions.",
                     description="",
                     theme=gr.themes.Soft(),
                     title="")

iface.launch()
