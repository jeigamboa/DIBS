### Preliminaries
import os
from dotenv import load_dotenv

# Import streamlit
import streamlit as st

# Import langchain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool
from langchain import hub

# Import supabase.db
from supabase.client import Client, create_client

### Main

# Load environment variables
file_path = 'C:/David/000 Work Prep and Independent Proj/' # Replace with path of DIBS on your system
load_dotenv(dotenv_path=file_path + "/DIBS/.env")  

# Initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initiating embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",)
 
# Initiating llm
llm = ChatOpenAI(model="gpt-4o-mini", 
                 max_completion_tokens=None,
                 temperature=0)

# Pulling prompt from hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs)
    return serialized, retrieved_docs

# Creating simulator tool



# Combining all tools
tools = [retrieve]

# Initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initiating streamlit app
st.set_page_config(page_title="Ask DIBS")
st.title("Ask DIBS")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Create the bar where we can type messages
user_question = st.chat_input("How may I help you?")

# Did the user submit a prompt?
if user_question:
    # Add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))
    # Invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})
    ai_message = result["output"]
    # Adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)
        st.session_state.messages.append(AIMessage(ai_message))
