import streamlit as st

from pdf_loader import (
    extract_text
)

from rag_engine import (
    chunk_text,
    create_vector_store,
    search_chunks,
    generate_answer
)

st.set_page_config(
    page_title="AI RAG Knowledge Base Assistant",
    page_icon="📚",
    layout="wide"
)

st.title(
    "📚 AI RAG Knowledge Base Assistant"
)

st.write(
    "Upload a PDF and ask questions about its contents."
)

pdf_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if pdf_file:

    text = extract_text(
        pdf_file
    )

    chunks = chunk_text(
        text
    )

    index, chunks = (
        create_vector_store(
            chunks
        )
    )

    st.success(
        f"{len(chunks)} chunks created successfully."
    )

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        retrieved_chunks = search_chunks(
            question,
            index,
            chunks
        )

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        st.subheader(
            "🤖 AI Answer"
        )

        st.success(
            answer
        )

        st.subheader(
            "📄 Retrieved Chunks"
        )

        for chunk in retrieved_chunks:

            st.info(
                chunk
            )