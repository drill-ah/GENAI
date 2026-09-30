import streamlit as st
from openai import OpenAIError

from resume_rag.documents import extract_resume_text
from resume_rag.generation import generate_resume
from resume_rag.retrieval import retrieve_context


st.set_page_config(page_title="Context-Aware Resume Generator", page_icon="📝")
st.title("Context-Aware Resume Generator")
st.write(
    "Tailor a resume to a job description using locally retrieved resume "
    "evidence and writing guidance."
)

with st.sidebar:
    st.header("Model settings")
    api_key = st.text_input("API key", type="password")
    model = st.text_input("Model", value="gpt-4o-mini")
    st.caption(
        "The API key is used for this session only. Resume content is sent to "
        "the model provider when you generate a draft."
    )

resume_file = st.file_uploader(
    "Upload your resume",
    type=("pdf", "docx", "txt"),
    help="PDF, DOCX, and plain text files are supported.",
)
job_description = st.text_area(
    "Target job description",
    height=240,
    placeholder="Paste the complete job description here...",
)

if st.button("Generate tailored resume", type="primary"):
    if resume_file is None:
        st.error("Upload a resume before generating.")
    elif not job_description.strip():
        st.error("Enter a job description before generating.")
    elif not api_key.strip():
        st.error("Enter an API key in the sidebar before generating.")
    else:
        try:
            resume_text = extract_resume_text(
                resume_file.name, resume_file.getvalue()
            )
            with st.spinner("Retrieving relevant context and generating draft..."):
                context = retrieve_context(resume_text, job_description)
                draft = generate_resume(
                    api_key,
                    model,
                    resume_text,
                    job_description,
                    context,
                )
        except (ValueError, UnicodeDecodeError) as error:
            st.error(str(error))
        except OpenAIError as error:
            st.error(f"The model provider returned an error: {error}")
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.markdown(draft)
            st.download_button(
                "Download draft as Markdown",
                data=draft,
                file_name="tailored_resume.md",
                mime="text/markdown",
            )
            with st.expander("Retrieved context"):
                for chunk in context:
                    st.markdown(
                        f"**{chunk.source}** · relevance {chunk.score:.3f}\n\n"
                        f"{chunk.text}"
                    )
