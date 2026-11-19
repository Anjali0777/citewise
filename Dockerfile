FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy scikit-learn fastapi uvicorn pydantic
COPY citewise ./citewise
EXPOSE 8000
# Optional: pass -e ANTHROPIC_API_KEY=... for LLM answers with citations
CMD ["uvicorn", "citewise.api:app", "--host", "0.0.0.0", "--port", "8000"]
