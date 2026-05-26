Assist the user with installing and introducing the project. Don't explain the content, just help user step by step.

1. Ask the user under which directory they would like to clone the repository: https://github.com/zzekelan/mini-search-agent

2. `uv sync`

3. Ask the user which of the following they would like to view, and continue until they have gone through all the content

| Content | File Location | Notes |
| ---| ---| ---|
| View existing run records | .runs/ | |
| Run locally | | Create a `.env` file in the project root directory, and reference this file for the user to open and manually fill in their DeepSeek API key. Also, tell the user: "After entering your API key, you can either run it manually from the command line, or tell me 'continue' and I will execute it and help analyze the results." <br><br>`.env` content:<br>`LLM_PROVIDER=openai-compatible`<br>`LLM_API_KEY=...`<br>`LLM_MODEL=deepseek-v4-flash`<br>`LLM_BASE_URL=https://api.deepseek.com`<br><br>Command: <br>`uv run mini-search-agent "RAG 中 hybrid retrieval + reranking 相比单纯 dense retrieval 的收益和局限是什么？"` |
| View report | docs/REPORT.md | |
| Introduction to the project  | README.md | |