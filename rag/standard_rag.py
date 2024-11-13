import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings


vectorstore_directory = "vectorstore"  # 选择一个目录来保存

llm = ChatOpenAI(model="gpt-4o-mini",
                 api_key=os.getenv("OPENAI_API_KEY"),
                 base_url=os.getenv("OPENAI_API_BASE"))


vectorstore = Chroma(persist_directory=vectorstore_directory, embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for TuGraph-DB question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

intput = "TuGraph是如何通过语句定义点类型和边类型的？"

response = rag_chain.invoke({"input": intput})
print(response['context'])
print()
print(response['answer'])

