# Context-Aware Resume Generator

A retrieval-augmented resume-writing demo. It extracts an existing resume,
retrieves relevant evidence and writing guidance for a target job, and sends
that context to an OpenAI chat model to produce a tailored draft.

## Project details

- **Student:** Shravan Kashyap
- **Roll number:** 23052428
- **Batch / Section:** B-1-IRT
- **Course:** Gen AI & Prompt Engineering
- **Trainer:** Sachin Sir
- **Submission date:** 30/09/2026
- **Domain:** Machine Learning / Generative AI

The generator is designed to preserve the candidate's facts: it must not add
skills, achievements, dates, credentials, or metrics that are not supported by
the source resume. Job requirements without supporting evidence should be
reported as gaps instead of being presented as qualifications.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a PDF, DOCX, or TXT resume, paste a job description, and provide an API
key in the sidebar. The key is held in the current Streamlit session and is not
written to the repository. The default model is `gpt-4o-mini`; change it in the
sidebar to match the model available to your API account.

## How it works

1. Extract text from the uploaded resume.
2. Split resume evidence and a small, built-in resume-writing guide into
   overlapping chunks.
3. Rank chunks against the job description with TF-IDF and cosine similarity.
4. Build a prompt containing the complete job description, the retrieved
   evidence, and explicit factuality constraints.
5. Generate a tailored resume draft and a separate list of gaps to address.

The retrieval stage is local. The uploaded resume, job description, and
retrieved context are sent to the configured model provider only when Generate
is clicked. Do not upload information you are not comfortable sharing with
that provider.

## Tests

```bash
python -m unittest discover -s tests -v
```
