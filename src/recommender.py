# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from src.prompt_template import get_anime_prompt

# class AnimeRecommender:
#     def __init__(self,retriever,api_key:str,model_name:str):
#         self.llm = ChatGroq(api_key=api_key,model=model_name,temperature=0)
#         self.prompt = get_anime_prompt()

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm = self.llm,
#             chain_type = "stuff",
#             retriever = retriever,
#             return_source_documents = True,
#             chain_type_kwargs = {"prompt":self.prompt}
#         )

#     def get_recommendation(self,query:str):
#         result = self.qa_chain({"query":query})
#         return result['result']
    

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.prompt_template import get_anime_prompt


class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        self.llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
        self.prompt = get_anime_prompt()

        # LCEL pipeline replaces RetrievalQA
        self.qa_chain = (
            RunnableParallel({"context": retriever | self._format_docs, "question": RunnablePassthrough()})
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        """Combine retrieved docs into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_recommendation(self, query: str):
        result = self.qa_chain.invoke(query)
        return result
