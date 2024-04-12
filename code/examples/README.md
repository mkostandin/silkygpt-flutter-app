

## GPT Index/LlamaIndex

A directory is filled with text and PDFs, and then a Vector Store Index is made using LlamaIndex and OpenAI API for retrieval. The Vector Store Index collects snippets of text from the documents in the directory and then embeds numerical values in them. These embeddings are compared to answer queries.

The following code produces a persist directory for the vector stores. 

``` python
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
```

## LangChain

In the above code, LangChain is used to create the chatbot, the LLM predictor. It combines the indexed information with the language model’s predictions to generate a coherent and contextually relevant response.

``` python
def chatbot(input_text):
    query_engine = new_index.as_query_engine()
    response = query_engine.query(input_text + "[ INSTRUCTIONS TO AI - OMITTED IN README FOR BREVITY]")
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
