

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

> 1. Use-Case Specific Chains: Chains can be thought of as assembling these components in particular ways in order to best accomplish a particular use case. These are intended to be a higher level interface through which people can easily get started with a specific use case. These chains are also designed to be customizable.

## OpenAI API - GPT-4
## Flutter - Dart Widget
