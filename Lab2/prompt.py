def build_prompt(context_chunks, user_query):
    context = "\n\n".join(context_chunks)
    return f"""You need to answer the question using only the context below.

    Context:
    {context}
    
    Question:
    {user_query}
    
    Answer:"""
