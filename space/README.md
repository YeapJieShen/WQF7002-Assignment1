---
title: Planning Delegate
emoji: 🗓️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
hf_oauth: true
# openid and profile are always granted; no extra scopes are requested (hf_oauth_scopes omitted on purpose).
pinned: false
license: mit
short_description: A planning agent that knows its limits (WQF7002 A1)
---

# Planning delegate

Chat with an agent that plans your week around travel time, opening hours, deadlines and your
own pace. It calls tools for every fact, asks before deleting anything and says when it does not
know.

Runs Qwen3-4B-Instruct-2507 on ZeroGPU, using the visitor's daily GPU quota; signing in with
Hugging Face gives a larger one. You can also point it at your own OpenAI-compatible https
endpoint for the session; the key is never stored.
Source and design notes: see the GitHub repository.
