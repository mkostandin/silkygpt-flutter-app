

## GPT Index/LlamaIndex

A directory is filled with text and PDFs, and then a Vector Store Index is made using LlamaIndex and OpenAI API for retrieval. The vector store collects snippets of text from the documents in the directory and then embeds numerical values in them. These embeddings are compared to answer queries.

The following code produces a GPT Index in the directory where text and PDFs are. 

``` python
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
```

## LangChain

In the above code, LangChain is used to create the chatbot, the LLM predictor. It combines the indexed information with the language model’s predictions to generate a coherent and contextually relevant response.

``` python
def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text + 'Respond helpfully. Don\'t say addicts. Alcoholics will always need to go to meetings. Alcoholics are never finished with the 12 Steps. Try to reference specific literature. ', response_mode="compact")
    return response.response
```
## Gradio

The main element of the UI is made using Gradio. 

``` python
iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Ask Silky:"),
                     outputs="text",
                     allow_flagging="never",
                     article="WARNING: This is a chatbot. It is wrong very often. It is not a substitute for professional medical advice, diagnosis, or treatment. If you think you are an alcoholic, get help first from real people, not a chatbot. If you have a medical emergency, please call 911 or your local emergency number. Help make Silky better. Email feedback to necyverse+silky@gmail.com",
                     description="",
                     theme=gr.themes.Soft(),
                     title="")

iface.launch()
```
