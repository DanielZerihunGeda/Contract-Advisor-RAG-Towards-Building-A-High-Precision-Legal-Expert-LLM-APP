"""

Flow :

    list of PDF, images, words --->. create parquet file of raw text

    parquet file of raw text ---> Send to paperQA tool
                             --->  Send to Storm tool
                             ---> send to tool XYZ


    we write the connector wrapper.


"""
if "import":
    import os, random, pandas as pd

    from groq import Groq, AsyncGroq
    from typing import Any, cast, List, Iterable, AsyncGenerator
    import asyncio, nest_asyncio
    from pydantic import Field
    from angle_emb import AnglE

    from utilmy import (pd_read_file, pd_to_file, log, glob_glob, json_save, json_load, date_now)


    from paperqa import Docs, Doc, LLMModel, EmbeddingModel, NumpyVectorStore
    from paperqa.llms import process_llm_config
    from paperqa.types import Text
    from paperqa.llms import VectorStore





####################################################################################
def run_qa(source: Any, question: str):
    """
    
         pyclean run_qa --source "ztmp/df_LZ_merge_90k.parquet" --question "summerize the document"

    """
    qa = PaperQA(source=source, question=question)
    log(qa.answer(question))



def create_indexes(dirin="ztmp/mypdf/**/*.*", dirout="ztmp/markit/"):
    """
        pip install markitdown
        from markitdown import MarkItDown
        from openai import OpenAI

        import os
        client = OpenAI(api_key="your-api-key-here")
        md = MarkItDown(llm_client=client, llm_model="gpt-4o-2024-11-20")
        supported_extensions = ('.pptx', '.docx', '.pdf', '.jpg', '.jpeg', '.png')


    """ 
    from markitdown import MarkItDown
    client = llm_client_get()

    md = MarkItDown(llm_client=client, llm_model="gpt-4o-2024-11-20")
    supported_extensions = ('.pptx', '.docx', '.pdf', '.jpg', '.jpeg', '.png')

    flist = glob_glob(dirin)
    flist = [f for f in flist if f.lower().endswith(supported_extensions)]
    df    = []
    for fi in flist:
        print(f"\nConverting {fi}...")
        try:
            result = md.convert(fi)
            txt    = result.text_content
            df.append([fi, txt])
        except Exception as e:
            print(f"Error converting {fi}: {str(e)}")

    df = pd.DataFrame(df, columns=["filename", "text"])

    ts = date_now(fmt="%y%m%d_%H%M%S") 
    pd_to_file(df, dirout + f"/df_doc_markdown_{ts}.parquet")


class test_storm:
    """To be compatible with STORM, the collection payload/metadata should have the following fields:
        - content: The main text content of the document.
        - title: The title of the document.
        - url: The URL of the document. STORM uses URL as the unique identifier of the document, so ensure different
          documents have different URLs.
        - description (optional): The description of the document.

        args: 
            qdrant_url: online url for your qdrant client
            qdrant_api: api_key
            collection_name: name of collection where the sources/embeddings stored 

     export QDRANT_URL="https://d62ceb93-002f-4fa4-ac0a-074f36eda8f8.us-east4-0.gcp.cloud.qdrant.io:6333"
     export QDRANT_API_KEY="8vJgp8xLJs1E_pF-a1R2gqU2HqtVYpsqHMDVNNHdIZYt6W9-BOst0g"

     pyclean test_storm run --topic "theory of evolution" -- max_token 1000

            output:
                Collection storm0 exists. Loading the collection...
                ***** Execution time *****
                run_knowledge_curation_module: 10.2629 seconds
                run_article_polishing_module: 0.5940 seconds
                ***** Token usage of language models: *****
                run_knowledge_curation_module

    """
    def __init__(self,
        max_tokens: int = 1500,
        model_low: str = 'llama3-8b-8192',
        model_best: str = 'llama3-70b-8192',
        
        embedding_model: str = 'all-MiniLM-L6-v2',

        vector_db_url: str = "",
        vector_db_api_key: str = "",

        output_dir: str = "./ztmp/storm/",

        max_conv_turn:   int = 1,
        max_perspective: int = 1,
        max_search_queries_per_turn: int = 1,
        max_thread_num: int = 1,
        k: int = 2,
        collection_name: str = "storm0"
    ):
        #custom class for llm client
        class Client(GroqModel):
          def __init__(self, model, api_key: str=None, **kwargs):
              super().__init__(model)
              self.client = llm_client_get(model=model)
              self.kwargs = kwargs

          def _create_completion(self, prompt: str, **kwargs):
              """Create a completion using the Groq API."""
              kwargs.pop("logprobs", None)
              kwargs.pop("logit_bias", None)
              kwargs.pop("top_logprobs", None)

              if kwargs.get("n", 1) != 1:
                  raise ValueError("Groq API only supports N=1")

              if kwargs.get("temperature", 1) == 0:
                  kwargs["temperature"] = 1e-8

              messages = [{"role": "user", "content": prompt}]
              response = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
              return json.loads(response.model_dump_json())

          def __call__(self, prompt: str, **kwargs) -> list[str]:
              """Call the Groq API to generate completions."""
              response = self._create_completion(prompt, **kwargs)
              self.log_usage(response)
              return [choice["message"]["content"] for choice in response["choices"]]


        from utilmy import os_makedirs
        os_makedirs(output_dir)

        from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
        from knowledge_storm.lm import OpenAIModel
        from knowledge_storm.rm import YouRM,VectorRM


        openai_kwargs = { 'temperature': 1.0, 'top_p': 0.9,}
        self.llm       = Client(model=model_low, max_tokens=max_tokens, **openai_kwargs)
        self.llm_best  = Client(model=model_best, max_tokens=max_tokens, **openai_kwargs)

        lm_configs = STORMWikiLMConfigs()
        lm_configs.set_conv_simulator_lm(self.llm)
        lm_configs.set_question_asker_lm(self.llm)
        lm_configs.set_outline_gen_lm(   self.llm)
        lm_configs.set_article_gen_lm(   self.llm)
        lm_configs.set_article_polish_lm(self.llm_best)

        
        self.engine_args = STORMWikiRunnerArguments(
            output_dir=output_dir,
            max_conv_turn=max_conv_turn,
            max_perspective=max_perspective,
            max_search_queries_per_turn=max_search_queries_per_turn,
            max_thread_num=max_thread_num
        )


        # Initialize retrieval module
        


        ######### Vector ############################################
        self.embedding_model = embedding_model
        vector_db_url     = os.environ.get("QDRANT_URL", vector_db_url )
        vector_db_api_key = os.environ.get("QDRANT_API_KEY", vector_db_api_key)

        ## self.rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=engine_args.search_top_k)
        self.rm = VectorRM(collection_name=collection_name, embedding_model=embedding_model, device="cpu", k=k)
        self.rm.init_online_vector_db(url=vector_db_url, api_key=vector_db_api_key)

        log("###### Runner ###############################")
        self.runner = STORMWikiRunner(self.engine_args, lm_configs, self.rm)


    def run(self, topic, do_research=True, do_outline=True, do_article=True, do_polish_article=True):
        self.runner.run( topic=topic,
            do_research=do_research,
            do_generate_outline=do_outline,
            do_generate_article=do_article,
            do_polish_article=do_polish_article,
        )
        self.runner.post_run()
        self.runner.summary()



def llm_client_get(service='groq', model='gpt-3.5-turbo', temperature=0.1):
    from openai import Groq

    if "gpt4" in model:
       from openai import openAI
       client = openAI(api_key= os.getenv("OPEN_API_KEY"))
       return client

    elif "llama" in model :
       log("using GROQ") 
       client = Groq(api_key= os.getenv("GROQ_API_KEY"))
       return client







###############################################################################################
class PaperQA:
    def __init__(self, source: Any, question: str):

        self.embedding_model: EmbeddingModel = AnglEEmbedding() # supports openai 
        self.texts_index = NumpyVectorStore(embedding_model=self.embedding_model)
        self.docs_index = NumpyVectorStore(embedding_model=self.embedding_model)
        self.source = source

        self.docname:  str = "Doc1"
        self.citation: str = "Sample Citation"
        self.dockey:   str = "12345"

        self.client    = LLMclient()
        self.llm_model = LLMModel_v1(config={"model": "llama3-70b-8192", "temperature": 0.1})

        self.texts = None
        self.doc = Doc(docname=self.docname, citation=self.citation, dockey=self.dockey)
        self.docs = Docs(llm_model=self.llm_model, client=self.client, embedding_client=self.client, texts_index=self.texts_index, docs_index=self.docs_index)

    def add_text(self):
        self.texts = text_objects(texts=self.source, doc=self.doc)
        return self.docs.add_texts(texts=self.texts, doc=self.doc)

    def add_path(self):
        pass

    def answer(self, question):
        self.question = question
        if self.add_text() is None and self.add_path() is None:
            raise ValueError("Invalid input")
        if self.add_text() is not None and self.add_path() is not None:
            raise ValueError("Invalid input")
        if self.add_text() is not None:
            self.add_text()
        else:
            self.add_path()
        return self.docs.query(self.question).answer()




#########################################################################################
##Embedding model
class AnglEEmbedding(EmbeddingModel):

    model: Any = Field(default=None)
    def __init__(self,    name: str = "WhereIsAI/UAE-Large-V1"):
        super().__init__()
        self.model = AnglE.from_pretrained(self.name, pooling_strategy='cls').cuda()

    async def embed_documents(self, client: Any, texts: List[str]) -> List[List[float]]:
        # encoding
        embeddings = self.model.encode(texts, to_numpy=True).tolist()
        return embeddings




  ###LLM model
class LLMModel_v1(LLMModel):
    config: dict = Field(default={"model": "llama3-70b-8192", "temperature": 0.1})
    name: str = "llama3-70b-8192"
    llm_type: str = "chat"
    async def achat(self, client: Any, messages: Iterable[dict[str, str]]) -> str:
        completion = await client.chat.completions.create(
            messages=messages, **process_llm_config(self.config)
        )
        return completion.choices[0].message.content or ""

    async def achat_iter(self, client: Any, messages: Iterable[dict[str, str]]) -> Any:
        """Return an async generator that yields chunks of the completion.
        I cannot get mypy to understand the override, so marked as Any
        """
        completion = await client.chat.completions.create(
            messages=messages, **process_llm_config(self.config), stream=True)
        async for chunk in cast(AsyncGenerator, completion):
            yield chunk.choices[0].delta.content


########################################################################
##### Async client #####################################################
def async_llm_client_get(api_key: str = os.getenv("GROQ_API_KEY"), model='llama-8b-8192', temperature=0.1):
    from groq import AsyncGroq
    client = AsyncGroq(api_key= api_key, model=model, temperature=temperature)
    return client




def text_objects(texts: Any, doc: Doc) -> List[Text]:
    from utils import pd_read_file
    if ".parquet" in texts:
        texts = pd_read_file(texts)
        text_li = texts['text'].tolist()
    elif isinstance(texts, list):
        text_li = texts
    else:
        raise TypeError("Invalid Data Type")

    text_ob = []
    for idx, text in enumerate(text_li):
        name = f"Chunk {idx + 1}"
        
        text_object = Text(text=text, name=name, doc=doc)
        text_ob.append(text_object)

    return text_ob




if __name__ == '__main__':
    import fire
    fire.Fire()

"""
##################################################################################
################################# Sample output ###################################
command: alias pyclean="python3 -u rag/PaperQA.py "
          "pyclean create_indexes"
output: 
    flist: ['ztmp/mypdf/notes_01.pdf']

    Converting ztmp/mypdf/notes_01.pdf...
    ztmp/markit//df_doc_markdown_250106_155349.parquet
    (1, 2)
################################################################################

command:  alias pyclean="python3 -u rag/PaperQA.py "
          "pyclean run_qa --source "ztmp/df_LZ_merge_90k.parquet" --question "summerize the document"
          
output: 
  The document discusses various topics related to technology and business. One excerpt highlights a deal for a lifetime 
  license to Microsoft Office Professional Plus 2019 for Windows or Mac, available for $25, which is 89% off the regular price (Chunk 13). 
  Another excerpt mentions FluxGen Sustainable Technologies' collaboration with Microsoft to implement AI and IoT-powered industrial water 
  management solutions in India, aiming to increase water efficiency and reduce dependence on freshwater supply (Chunk 46). Additionally, 
  a message blocks access to a website from countries within the European Economic Area due to the enforcement of the General Data Protection 
  Regulation (GDPR) (Chunk 62). Furthermore, Ola Electric is accused of stealing data from MapMyIndia to develop its Ola Maps interface (Chunk 8). 
  Lastly, an author discusses their investment strategy amidst market volatility, considering smaller tech stocks and bitcoin (Chunk 64).

cmd: "pyclean test_storm run --topic "theory of evolution"

output:
    ollection storm0 exists. Loading the collection...
    ###### Runner ###############################
    ***** Execution time *****
    run_knowledge_curation_module: 7.8994 seconds
    run_outline_generation_module: 0.8609 seconds
    run_article_generation_module: 4.4867 seconds
    run_article_polishing_module: 0.6887 seconds
    ***** Token usage of language models: *****
    run_knowledge_curation_module
        llama3-8b-8192: {'prompt_tokens': 3633, 'completion_tokens': 2451}
        llama3-70-8192: {'prompt_tokens': 0, 'completion_tokens': 0}
    run_outline_generation_module
        llama3-8b-8192: {'prompt_tokens': 1291, 'completion_tokens': 485}
        llama3-70-8192: {'prompt_tokens': 0, 'completion_tokens': 0}
    run_article_generation_module
        llama3-8b-8192: {'prompt_tokens': 1484, 'completion_tokens': 2163}
        llama3-70-8192: {'prompt_tokens': 0, 'completion_tokens': 0}
    run_article_polishing_module
        llama3-8b-8192: {'prompt_tokens': 2275, 'completion_tokens': 264}
        llama3-70-8192: {'prompt_tokens': 0, 'completion_tokens': 0}
    ***** Number of queries of retrieval models: *****
    run_knowledge_curation_module: {'VectorRM': 2}
    run_outline_generation_module: {'VectorRM': 0}
    run_article_generation_module: {'VectorRM': 0}
    run_article_polishing_module: {'VectorRM': 0}
    
"""