

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
> ### Overview
>
> GPT Index is a powerful tool that connects large language models (LLMs) with external data sources. It allows you to access and use any information you need without worrying about the prompt size limit of LLMs. GPT Index also simplifies the code and reduces the cost of building LLM applications.
>
> ### Features
>
> With GPT Index, you can:
>
> - Use different types of index data structures to store and retrieve your data efficiently and flexibly.
 > - Connect your data sources with LLMs easily using built-in connectors for Google Docs, Slack, and more.
> - Optimize the performance and cost of your LLM applications using transparent metrics and tools.
> - Perform various tasks with LLMs using general-purpose queries, such as:
>     - Answering questions
>     - Summarizing texts
>     - Generating texts (stories, tasks, TODOs, emails, etc.)
>
> GPT Index is designed to help you unleash the full potential of LLMs for knowledge generation and reasoning.

## LangChain
https://docs.langchain.com/docs/

>LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an api, but will also:
>
> 1. Be data-aware: connect a language model to other sources of data
> 1. Be agentic: Allow a language model to interact with its environment
> 1. As such, the LangChain framework is designed with the objective in mind to enable those types of applications.
>
> There are two main value props the LangChain framework provides:
>
> 1. Components: LangChain provides modular abstractions for the components neccessary to work with language models. LangChain also has collections of implementations for all these abstractions. The components are designed to be easy to use, regardless of whether you are using the rest of the LangChain framework or not.
> 1. Use-Case Specific Chains: Chains can be thought of as assembling these components in particular ways in order to best accomplish a particular use case. These are intended to be a higher level interface through which people can easily get started with a specific use case. These chains are also designed to be customizable.

## OpenAI API - GPT-4
## Flutter - Dart Widget
