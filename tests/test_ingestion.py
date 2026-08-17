from rag.ingestion import extract_text_from_pdf


text = extract_text_from_pdf(
    r"C:\\Users\\piotr\\Desktop\\llm-chatbot\\data\\EN-GL30MEU-Datasheet-250825-1-1.pdf"
)

print(text)