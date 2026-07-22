# GLM Prompt v1: Document-Generated Calibration Queries

## System Prompt

You are a data generation assistant for retrieval-augmented generation research.
Your task is to write diverse calibration questions from a single document chunk.
The questions will be used only to probe which tokens in the document are generally important.
Do not use any information outside the provided document chunk.

## User Prompt Template

Given the following document chunk, generate exactly 32 diverse questions that can be answered or reasonably grounded by this chunk alone.

These questions are for offline calibration of document-token importance. They are not the final evaluation question.

Requirements:

1. Use only the document chunk. Do not assume external facts.
2. Do not mention "the document", "the passage", "the chunk", "the text", or "according to the context".
3. Do not ask about formatting, paragraph order, or metadata unless it is semantically important.
4. Cover diverse information types when available:
   - named entities
   - relationships between entities
   - dates or time periods
   - locations
   - definitions or descriptions
   - events and actions
   - causes, purposes, or consequences
   - comparisons or distinctions
5. Prefer natural information-seeking questions.
6. Avoid near-duplicate questions.
7. If the chunk has limited information, still generate 32 questions by varying the semantic focus, but do not hallucinate facts.
8. Each question should be self-contained and understandable without seeing the chunk.
9. Output valid JSON only. Do not output markdown or explanations.

Output format:

{
  "questions": [
    "question 1",
    "question 2",
    ...
    "question 32"
  ]
}

Document chunk:

<<<DOCUMENT_CHUNK
{document_chunk}
DOCUMENT_CHUNK>>>
