from pathlib import Path
from ollama import chat
import json

MODEL = "qwen3:8b"


question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""


## WRITE ##
## Move useful information out of temporary model context and into
## persistent external state, so later calls can reuse it.
service_status = {
    "wifi": "operational"
}

state = {
    "issue": "wifi_connection",
    "device": "Windows laptop",
    "password_recently_changed": True,
    "wifi_service_status": "operational",
    "service_status_checked": True
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

with open("state.json", "r") as file:
    state = json.load(file)

print(state)


## SELECT CONTEXT FILES BASED ON QUESTION
## Create the function that takes the student's question, takes some keywords and chooses the relevant files from the knowledge base. Return a list of the selected files.
## For example, if the question has the keyword "print" or "printer", then the function should return the file "knowledge/printing.txt" in a list.
def select_context(question):
    keyword_files = {
        "wi-fi": ["knowledge/wifi_setup.txt", "knowledge/service_status.txt"],
        "wifi": ["knowledge/wifi_setup.txt", "knowledge/service_status.txt"],
        "password": ["knowledge/password_changes.txt"],
        "print": ["knowledge/printing.txt"],
        "printer": ["knowledge/printing.txt"],
        "vpn": ["knowledge/vpn.txt"],
        "email": ["knowledge/email_setup.txt"],
        "projector": ["knowledge/classroom_projectors.txt"],
    }

    selected = []

    lower = question.lower()

    for keyword, files in keyword_files.items():
        if keyword in lower:
            for file in files:
                if file not in selected:
                    selected.append(file)

    return selected


selected_files = select_context(question)

## READ SELECTED FILES and add their contents to the context variable.
context = ""

for file in selected_files:
    context += Path(file).read_text()
    context += "\n\n"


## COMPRESS CONTEXT
## Add logic to compress the context from above by calling Qwen with "context" and the "question" as the parameter
## The response from Qwen should be the compressed context. Store it in a variable called "compressed_context"
def compress_context(context, question):
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Extract only the information that is relevant to the user's problem.

Do not solve the problem.
Do not add new information.
"""
            },
            {
                "role": "user",
                "content": f"""
USER PROBLEM:

{question}

DOCUMENT:

{context}
"""
            }
        ]
    )

    return response.message.content


compressed_context = compress_context(
    context,
    question
)

## Print the length of the compressed context
print("Compressed context characters:", len(compressed_context))

## Now, call Qwen again with the compressed context and the student's question. Store the response in a variable called "response" and print the response from Qwen.
## Ensure the model produces a structured output
response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": """
You are a university IT support assistant.

Answer using only the information provided.

You must respond ONLY with a single, valid JSON object matching the
schema below. Do not include any conversational filler, markdown
formatting blocks (like ```json), or extra text.

JSON Schema:
{
  "issue": "string description",
  "device": "string description",
  "cause": "string description",
  "steps": ["step 1", "step 2", "..."]
}
"""
        },
        {
            "role": "user",
            "content": f"""
QUESTION:

{question}

AVAILABLE INFORMATION:

{compressed_context}
"""
        }
    ]
)

print(response.message.content)

## WRITE the above output in an artifact called "state".
## The model already returned valid JSON text, so save it directly.
with open("state.json", "w") as file:
    file.write(response.message.content)

## Update the rest of the code so that it uses the "state" artifact as part of the context.
## It is important to ensure that the model uses only the relevant parts from the "state" artifact and not the entire artifact.
## For this, you may have to think of a good structure for the "state" artifact and how to use it in the context.

## Read the state artifact back as a Python dictionary.
with open("state.json", "r") as file:
    state = json.load(file)

## ISOLATE ##
## Keep different contexts separated. This diagnostic task only needs the
## fields relevant to it, pulled out of the state artifact explicitly.
diagnostic_context = {
    "problem": question,
    "device": state["device"],
    "cause": state["cause"],
    "steps": state["steps"],
    "wifi_status": service_status["wifi"],
}

## A different task (e.g. an administrator's weekly report) would use a
## different context and must NOT be mixed in here.
report_context = {
    "total_wifi_cases": 37,
    "resolved_cases": 29,
    "unresolved_cases": 8,
}

final_response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a university IT support assistant."
        },
        {
            "role": "user",
            "content": f"""
PROBLEM:

{diagnostic_context["problem"]}

DEVICE:

{diagnostic_context["device"]}

WI-FI SERVICE STATUS:

{diagnostic_context["wifi_status"]}

DIAGNOSIS:

{diagnostic_context["cause"]}

RECOMMENDED STEPS:

{diagnostic_context["steps"]}
"""
        }
    ]
)

print(final_response.message.content)
