from litellm import completion

def llm_call(messages, model="gpt-3.5-turbo"):
    response = completion(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content