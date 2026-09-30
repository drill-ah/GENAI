from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


WRITING_GUIDANCE = (
    "Use concise, specific bullet points that begin with strong action verbs.",
    "Prioritize experience and skills relevant to the target role; retain "
    "accurate titles, employers, dates, and qualifications.",
    "Use measurable outcomes only when the candidate's source material "
    "provides those metrics.",
    "Use standard section headings and clear, applicant-tracking-system "
    "friendly wording.",
    "Do not claim a required skill, certification, responsibility, or result "
    "unless the candidate's evidence supports it.",
)


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


def _split_into_chunks(text: str, max_words: int = 140, overlap: int = 24) -> list[str]:
    words = text.split()
    if not words:
        return []
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")

    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks


def retrieve_context(
    resume_text: str,
    job_description: str,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """Retrieve resume evidence and guidance relevant to the target role."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    chunks = [
        RetrievedChunk(source="Resume evidence", text=chunk, score=0.0)
        for chunk in _split_into_chunks(resume_text)
    ]
    chunks.extend(
        RetrievedChunk(source="Writing guidance", text=guide, score=0.0)
        for guide in WRITING_GUIDANCE
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    document_vectors = vectorizer.fit_transform(chunk.text for chunk in chunks)
    query_vector = vectorizer.transform([job_description])
    scores = cosine_similarity(query_vector, document_vectors).ravel()

    ranked = sorted(
        (
            RetrievedChunk(chunk.source, chunk.text, float(scores[index]))
            for index, chunk in enumerate(chunks)
        ),
        key=lambda chunk: chunk.score,
        reverse=True,
    )
    return ranked[:top_k]
