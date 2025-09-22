🧠 The Data Company – AI for Mental Health Chatbot

This repository contains the prototype submission for Google Hack2Skill AI for Mental Health Hackathon.
The project demonstrates an AI-powered mental health chatbot that combines LLM-based conversation, sentiment & emotion detection, aspect analysis, crisis safety checks, and RAG (Retrieval-Augmented Generation) into one pipeline.

The goal is to create an empathetic, safe, and context-aware AI companion for mental health support.

🚀 Features

Conversational AI (Phi-3 Mini with fallback) – Generates empathetic responses.

Sentiment & Emotion Detection – Identifies user’s feelings (positive/negative/neutral, sadness, anxiety, anger, etc.).

Aspect-Based Sentiment Analysis (ABSA) – Detects key aspects like study, job, relationships.

Crisis Detection & Safety Filter – Prevents unsafe replies and provides helplines.

RAG Knowledge Base – Retrieves coping tips and supportive content.

Multimodal Inputs – Supports text, voice, and facial emotion.

📂 Repository Structure

The_Data_Company_Chatbot.ipynb
Main Jupyter Notebook with the end-to-end chatbot pipeline (LLM + Sentiment + Emotion + ABSA + Safety + RAG).

The_Data_Company_Chatbot.ipynb – Colab.pdf
Exported PDF of the notebook for quick viewing without execution.

RAG_Knowledge_Base_WithID.xlsx
Knowledge base file used for Retrieval-Augmented Generation (RAG).
Contains mental health tips, coping strategies, and supportive responses mapped with IDs.

Team-Name-The-Data-Company.pdf
📑 Hackathon presentation deck — explains the problem, solution, features, process flow, technologies, and impact.

Meeting with (Postgrad C00313606) Mitul Srivastava.pdf
🎥 Video recording file demonstrating the working prototype demo for hackathon submission.

🏗️ How to Run

Open The_Data_Company_Chatbot.ipynb in Google Colab or Jupyter Notebook.

Install dependencies:

pip install torch transformers sentencepiece accelerate bitsandbytes
pip install sentence-transformers faiss-cpu pandas gradio soundfile librosa gTTS


Mount Google Drive and ensure RAG_Knowledge_Base_WithID.xlsx is accessible.

Run the notebook cells in order.

Start the chat loop (Cell 15) → choose text, voice, or facial+voice mode.

📊 Example Output

Input:
I am stressed about exams.

Output:

Bot Reply: "I understand exam pressure can be overwhelming. Try breaking tasks into smaller steps and practicing breathing exercises."

Sentiment: Negative

Emotion: Anxiety (0.82 confidence)

Aspects: Study → negative

🎯 USP of the Solution

Unlike generic chatbots, this system integrates:

LLM conversation

Emotion & Aspect understanding

Crisis safety filters

Knowledge retrieval

Multimodal input

💡 Making it empathetic, safe, and context-aware – uniquely designed for responsible AI in mental health.
