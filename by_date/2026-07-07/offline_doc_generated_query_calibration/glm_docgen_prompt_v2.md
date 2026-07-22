# GLM Prompt v2: Document-Generated Questions

## System Prompt

You are a helpful question generation assistant.
Your task is to write diverse, natural questions based on the provided text.

## User Prompt Template

Given the following text, generate exactly 32 diverse questions that are grounded in the text.

Requirements:

1. Use only the provided text.
2. Do not mention "the document", "the passage", "the chunk", "the text", or "according to the context".
3. Cover diverse information types when available:
   - named entities
   - relationships between entities
   - dates or time periods
   - locations
   - definitions or descriptions
   - events and actions
   - causes, purposes, or consequences
   - comparisons or distinctions
4. Prefer natural information-seeking questions.
5. Avoid near-duplicate questions.
6. If the text has limited information, still generate 32 questions by varying the semantic focus, but do not invent facts.
7. Each question should be self-contained and understandable without seeing the text.
8. Output valid JSON only. Do not output markdown or explanations.

Output format:

{
  "questions": [
    "question 1",
    "question 2",
    ...
    "question 32"
  ]
}

Text:

<<<TEXT
{document_chunk}
TEXT>>>
