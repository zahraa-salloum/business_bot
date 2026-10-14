---
title: Second Spark Assistant
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# Second Spark Business Assistant

EECE 503P/798S Agentic Systems, C3 Introduction to Agents

Zahraa Salloum, 202674524

An agent that represents Second Spark, a fictional electronics repair and refurbishing workshop in Beirut. The agent is called Watt. It answers questions from the business files, routes every message to the right path, collects leads and logs feedback through tool calls.

## What Is Inside

1. `me/about_business.pdf` and `me/business_summary.txt`: the business identity.
2. `business_agent.ipynb`: the main notebook with the persona, the instructions, the router, the two tools, the agent loop, the reflection step and the Gradio interface.
3. `app.py`: the same code without the demonstration cells, for deployment.
4. `requirements.txt`: the libraries used.
5. `.env.example`: the template for the API key.

## How to Run

1. Install the libraries with `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and add a `GEMINI_API_KEY` or an `OPENAI_API_KEY`.
3. Open `business_agent.ipynb` and run all cells, or run `python app.py` to start only the chat interface.

Leads, feedback and the reflection drafts are printed and also saved inside the `logs` folder.
