from openai import OpenAI

from resume_rag.retrieval import RetrievedChunk


def build_prompt(
    resume_text: str,
    job_description: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    evidence = "\n\n".join(
        f"[{chunk.source}; relevance={chunk.score:.3f}]\n{chunk.text}"
        for chunk in retrieved_chunks
    )
    return f"""You are a careful resume-writing assistant. Tailor the candidate's
resume to the target role using the supplied source material.

Factuality rules:
- Treat the source resume and retrieved resume evidence as the only source of
  truth about the candidate.
- Do not invent or infer skills, employers, job titles, dates, education,
  certifications, responsibilities, achievements, or numerical results.
- You may reorganize and rephrase supported information, but preserve its
  meaning and factual details.
- Do not state that the candidate meets a requirement unless the evidence
  supports it. List unsupported or unclear requirements under "Gaps to Address".
- Do not include private contact details unless they appear in the resume.

Return exactly these sections:
## Tailored Resume
## Gaps to Address

Target job description:
{job_description}

Full source resume:
{resume_text}

Retrieved context:
{evidence}
"""


def generate_resume(
    api_key: str,
    model: str,
    resume_text: str,
    job_description: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    if not api_key.strip():
        raise ValueError("An API key is required to generate a resume.")
    if not model.strip():
        raise ValueError("A model name is required.")

    client = OpenAI(api_key=api_key.strip())
    response = client.chat.completions.create(
        model=model.strip(),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Produce accurate, evidence-grounded resume drafts. "
                    "Never fabricate candidate qualifications."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(
                    resume_text, job_description, retrieved_chunks
                ),
            },
        ],
    )
    result = response.choices[0].message.content
    if not result or not result.strip():
        raise RuntimeError("The model returned an empty response.")
    return result.strip()
