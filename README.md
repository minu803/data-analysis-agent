# Supply Chain AI Agent
An interactive Streamlit app designed to analyze supply chain datasets using a modular AI agent architecture powered by OpenAI and PandasAI.

---

## Project Background & Motivation

I recently became interested in the concept of AI agents and wanted to explore how they could be applied to real-world workflows—especially those involving repetitive data analysis tasks. As someone working with data frequently, I saw the potential for an agent to boost my day-to-day efficiency.

This project is both a learning exercise and a practical tool. It gave me the opportunity to deepen my understanding by building a functioning agent system from scratch.

My architectural inspiration came from [Anthropic’s “Building Effective Agents”](https://www.anthropic.com/engineering/building-effective-agents), specifically the **prompt chaining** pattern. This approach breaks down complex tasks into sequential steps, where each step is validated before passing to the next—ensuring both transparency and modularity.

---

## Project Overview

The current system includes **three main agent components** structured in a sequential flow:

1. **File Processing Agent**: Handles file upload and parsing (.csv and Excel).
2. **Data Analysis Agent**: Interprets user questions and analyzes the dataset using PandasAI.
3. **Domain Expert Agent**: Uses GPT-4 to generate structured expert insights in the context of supply chain analytics.

All interactions are wrapped in a chat interface built with Streamlit, allowing users to upload a dataset and chat naturally to extract insights.

---

## Features

- Upload CSV or Excel files
- Ask natural language questions about your data
- Visualizations and tables generated dynamically
- Supply chain expert feedback with recommendations and confidence scores
- Analysis history with timestamped insights

---

## How It Works

1. **Setup**: Store your OpenAI API key in a `.env` file as `OPENAI_API_KEY`.
2. **Run**: Launch the app with:

   ```bash
   streamlit run streamlit_app.py
   ```
3. Upload: Choose your supply chain dataset (CSV or Excel).
4. Chat: Start asking questions such as:
- “What are the top-selling products?”
- “Are there any inventory shortages?”
- “How does lead time affect revenue?”
- “What trends do you notice in defect rates?”

## Future Work 
This is an early prototype and there are several areas I'd like to explore further:

- **Moving beyond PandasAI**: While PandasAI made it easy to get started, I’m interested in more robust ways to implement an LLM-powered data agent without relying on external abstractions.
- **Enabling iterative refinement**: Currently, the agents perform single-turn analysis. I’d like to explore architectures that support multi-turn or looped reasoning.
- **Agent architecture improvements**: I’m curious about best practices for designing scalable, composable agent systems, especially when chaining domain-specific tools.

## 🙏 Acknowledgments

- Inspired by Anthropic’s [“Building Effective Agents”](https://www.anthropic.com/engineering/building-effective-agents)
- Built using:
  - [PandasAI](https://github.com/gventuri/pandas-ai)
  - [OpenAI API](https://platform.openai.com/)
  - [Streamlit](https://streamlit.io)
   
