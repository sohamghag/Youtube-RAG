
import streamlit as st
from supporting_function import (
    extract_video_id,
    extract_important_points,
    language_converter,
    get_transcript, make_notes,create_chunks,rag_answer,create_embeddings
)



# Page configuration with modern theme
st.set_page_config(
    page_title="VidSynth AI - YouTube RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("🎬 VidSynth AI")
    st.markdown("---")
    st.markdown("Transform any YouTube video into key topics, a podcast, or a chatbot.")
    st.markdown("### Input Details")

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    language = st.text_input("Video Language Code", placeholder="e.g., en, hi, es, fr", value="en")

    task_option = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes For You"]
    )

    submit_button = st.button("✨ Start Processing")
    st.markdown("---")

st.markdown("<h1 style=text-align:center>Youtube RAG Application</h1>",unsafe_allow_html=True)


if submit_button:
    if youtube_url and language:
        video_id=extract_video_id(youtube_url)
        if video_id:
            with st.spinner("Step 1/3 : Fetching Transcript....."):
                video_transcript=get_transcript(video_id,language)

                if language!="en":
                    with st.spinner("Step 1.5/3 : Translating Transcript into English, This may take few moments......"):
                        video_transcript=language_converter(video_transcript)

        if task_option == "Notes For You":
                    with st.spinner("Step 2/3: Extracting important Topics..."):
                        imp_points = extract_important_points(video_transcript)
                        st.subheader("Important Topics")
                        st.write(imp_points)
                        st.markdown("---")

                    with st.spinner("Step 3/3 : Generating Notes for you."):
                        notes=make_notes(video_transcript)
                        st.subheader("Notes for you")
                        st.write(notes)
                    st.success("Summary and Notes Generated.")

        if task_option == "Chat with Video":
                    with st.spinner("Step 2/3 : Creating Chunks And Embedding..."):
                        video_chunks=create_chunks(video_transcript)
                    with st.spinner("Step 2.5/3 : Storing Information in Vector Database..."):
                        vectorstore = create_embeddings(video_chunks)
                        st.session_state.vector_store = vectorstore
                        st.success('Video is ready for chat.....')
                    st.session_state.messages = []


    # chatbot session

if task_option == "Chat with Video" and "vector_store" in st.session_state:
        st.divider()
        st.subheader("Chat with Video")

        # Display the entire history
        for message in st.session_state.get('messages', []):
            with st.chat_message(message['role']):
                st.write(message['content'])

        # user_input
        prompt = st.chat_input("Ask me anything about the video.")
        if prompt:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.write(prompt)

            with st.chat_message('assistant'):
                response = rag_answer(prompt, st.session_state.vector_store)
                st.write(response)

            st.session_state.messages.append({'role': 'assistant', 'content': response})

