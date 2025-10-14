import re
import shutil
import time
import streamlit as st
from dotenv import load_dotenv
from googleapiclient import model
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

import os

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        print(match.group(1))
        return match.group(1)
    st.error("Invalid URL")
    return None

def get_transcript(url_id,language):
    ytt=YouTubeTranscriptApi(
        proxy_config=WebshareProxyConfig(
            proxy_username="uovoumor",
            proxy_password="udk28jufv401",
        )
    )
    try:
        print("Inside get_transcript function")
        transcript=ytt.fetch(url_id,languages=[language])
        print("Transcript Fetched Successfully",transcript)
        full_transcript=" ".join([t.text for t in transcript])
        time.sleep(10)
        return full_transcript
    except Exception as e:
        print(f"Invalid Url {e}")

def language_converter(transcript):
    try:
        model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key,temperature=0.2)
        prompt=PromptTemplate(
            template="""
           You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker’s voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
            """,
            input_variables=['transcript'],
        )
        chain=prompt | model
        result=chain.invoke({"transcript":transcript})
        print(result)
        return result.content
    except Exception as e:
        print(f"Invalid Language Converter Error {e}")

def extract_important_points(transcript):
    try:
        model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key,temperature=0.2)
        prompt=PromptTemplate(
            template="""
              You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
            """,
            input_variables=['transcript'],
        )
        chain=prompt | model
        result=chain.invoke({"transcript":transcript})
        print(result)
        return result.content
    except Exception as e:
        print(f"Invalid Language Converter Error {e}")

def make_notes(transcript):
    try:
        model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key,temperature=0.2)
        prompt=PromptTemplate(
            template="""
            You are an AI note-taker. Your task is to read the following YouTube video transcript 
                and produce well-structured, concise notes.

                ⚡ Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.

                Here is the transcript:
                {transcript}""",
            input_variables=['transcript'],
        )
        chain=prompt | model
        result=chain.invoke({"transcript":transcript})
        print(result)
        return result.content
    except Exception as e:
        print(f"Invalid Language Converter Error {e}")


def create_chunks(transcript):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    text_documents=text_splitter.create_documents([transcript])
    return text_documents

def create_embeddings(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", transport="grpc")
    vector_store = Chroma.from_documents(docs, embeddings)  # No persist_directory → in-memory
    return vector_store


def rag_answer(question,vector_store):
    result=vector_store.similarity_search(question,k=4)
    context=""
    for i in result:
        context=f"{context}+{i.page_content} /n"

    prompt=PromptTemplate(
        template="""
        You are a kind, polite, and precise assistant.
                - Begin with a warm and respectful greeting (avoid repeating greetings every turn).
                - Understand the user’s intent even with typos or grammatical mistakes.
                - Answer ONLY using the retrieved context.
                - If answer not in context, say:
                  "I couldn’t find that information in the database. Could you please rephrase or ask something else?"
                - Keep answers clear, concise, and friendly.

                Context:
                {context}

                User Question:
                {question}

                Answer:
        """,
        input_variables=["context","question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.2)
    chain=prompt | model
    result=chain.invoke({"context":context,"question":question})
    print(result)
    return result.content











# without langchain

# def language_converter():
#         client=genai.Client(api_key=api_key)
#         result=client.models.generate_content(model="gemini-2.5-flash",contents=[video_transcript,"Translate into english"])
#         print(result.text)
#         return result.text

# video_transcript_english=None

# if language!="en":
#  video_transcript_english=language_converter()
# print(video_transcript_english)


