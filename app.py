import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import graphviz
from gtts import gTTS
import os
import tempfile

BASE_MODEL_PATH = "./base_model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    return_dict=True,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device)


st.title("Smart LMS using AI")
st.markdown("Ask a question, get a summary, or extract key concepts.")


task = st.selectbox("Select Task", ["Question Answering", "Topic Summarization", "Generate Mindmap"])


user_input = st.text_area("Enter your text:")
submit = st.button("Submit")

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=300,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_mindmap(key_concepts_json):
    try:
        concepts = json.loads(key_concepts_json)
        graph = graphviz.Digraph(format='png')

        def add_nodes_and_edges(node, parent=None):
            graph.node(node["name"])
            if parent:
                graph.edge(parent, node["name"])
            for child in node.get("children", []):
                add_nodes_and_edges(child, node["name"])

        add_nodes_and_edges(concepts)
        out_path = tempfile.mktemp(suffix=".png")
        graph.render(filename=out_path, cleanup=True)
        return out_path + ".png"
    except Exception as e:
        return f"Error generating mind map: {e}"

def text_to_speech(response_text):
    tts = gTTS(text=response_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        return temp_audio_file.name


if submit and user_input:
    with st.spinner("Generating response..."):
        if task == "Question Answering":
            response = generate_response(f"Answering the following question: {user_input}")
            st.markdown(f"**Answer:** {response}")
            audio_path = text_to_speech(response)
            st.audio(audio_path)

        elif task == "Topic Summarization":
            response = generate_response(f"Summarize the following topic: {user_input}")
            st.markdown(f"**Summary:** {response}")
            audio_path = text_to_speech(response)
            st.audio(audio_path)

        elif task == "Generate Mindmap":
            response = generate_response(
                f"Generated Mindmap: {user_input}"
            )
            st.markdown(f"**Key Concepts (JSON):** {response}")
            mindmap_path = generate_mindmap(response)
            if os.path.exists(mindmap_path):
                st.image(mindmap_path, caption="Mindmap of Key Concepts")
            else:
                st.error(mindmap_path)

            audio_path = text_to_speech(response)
            st.audio(audio_path)
