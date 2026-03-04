from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.core.logger import logger
from src.core.settings import settings


class AIServices:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b", api_key=settings.GROQ_API_KEY, temperature=0.7
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are an intelligent AI assistant that helps answer questions based on provided context. Use the following context to answer the question. If you don't know the answer, say you don't know. 
            
            If the question is not related to the context, politely respond that you are tuned to only answer questions related to the provided context.
            
            And if the question is related to the context, provide a concise and accurate answer based on the information in the context. Always use all available information in the context to provide the best possible answer and add citations in the format [source1], [source2], etc. for any information used from the context.
            
            Context: {context}
            
            Query: {query}
            """
        )
        logger.info("AIServices initialized successfully")
    
    def generate_answer(self, query: str, context: str) -> str:
        chain = self.prompt_template | self.llm
        response = chain.invoke({"query": query, "context": context}).model_dump()
        
        return response