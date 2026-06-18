import os
import requests
from embedder import search_chunks
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_answer(question: str, filename: str) -> dict:
    try:
        # Step 1: Search ChromaDB for relevant chunks
        print(f"🔍 Searching for: {question}")
        chunks = search_chunks(question, filename, top_k=5)

        if not chunks:
            return {
                "answer": "This information is not available in the provided documents.",
                "sources": []
            }

        # Step 2: Build context from chunks
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"\n[Source {i+1} - Page {chunk['page']}, Type: {chunk['type']}]\n"
            context += chunk["text"] + "\n"

        # Step 3: Call Groq API
        print("🤖 Calling Groq AI...")

        prompt = f"""You are a helpful document assistant.
Answer the user's question using ONLY the context provided below.
Always answer in plain text sentences only.
Never use bullet points, tables, or markdown formatting.
If the answer is not clearly in the context, say: "This information is not available in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
        )

        data = response.json()
        print(f"🔍 Groq raw response: {data}")

        answer = data["choices"][0]["message"]["content"].strip()

        # Step 4: Format sources
        sources = []
        for chunk in chunks:
            sources.append({
                "page": chunk["page"],
                "type": chunk["type"],
                "preview": chunk["text"][:100]
            })

        print(f"✅ Answer generated!")
        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "answer": "An error occurred while processing your question.",
            "sources": []
        }


# ─── TEST ─────────────────────────────────────────
if __name__ == "__main__":
    from embedder import store_chunks
    from parser import parse_pdf
    import os

    test_file = "uploads/test.pdf"

    if os.path.exists(test_file):
        print("📄 Parsing and storing PDF...")
        chunks = parse_pdf(test_file)
        store_chunks(chunks, "test.pdf")

        question = "What is this document about?"
        print(f"\n❓ Question: {question}")

        result = get_answer(question, "test.pdf")

        print(f"\n💬 Answer: {result['answer']}")
        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  - Page {s['page']} ({s['type']}): {s['preview']}...")
    else:
        print("❌ No test PDF found.")