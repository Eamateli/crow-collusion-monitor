import os

from openai import OpenAI

client = OpenAI(
    base_url="https://e-a-mateli--ep-crow-trusted-server.us-west.modal.direct/v1",
    api_key=f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}",
)

completion = client.chat.completions.create(
    model="nvidia/GLM-5.3-Flash-NVFP4",
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical assistant.",
        },
        {
            "role": "user",
            "content": "What is the weather in Paris?",
        },
    ],
    temperature=0.3,
    max_tokens=2048,
    top_p=0.9,
    stream=False,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
    ],
    tool_choice="auto",
    extra_body={"reasoning": {"enabled": True}},
    extra_headers={
        "Modal-Routing-Affinity-Key": os.environ["MODAL_ROUTING_AFFINITY_KEY"],
    },
)
print(completion.choices[0].message.content)