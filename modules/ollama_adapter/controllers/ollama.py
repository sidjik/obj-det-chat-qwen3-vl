#import ollama
import os
from ollama import Client
from ollama._types import ChatResponse
from pathlib import Path
from pydantic import BaseModel, Field
import numpy as np
from typing import Any, Callable



ollama = Client(
    host = os.environ.get("OLLAMA_HOST", "localhost:11434"),
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')} \
        if 'ollama.com' in str(os.environ.get("OLLAMA_HOST")) else None

)


def answer(messages: list[dict[str, str]], model: str, options: dict[str, Any] | None) -> dict[str, str]:
    kwargs = dict() 
    if options is not None:
        kwargs['options'] = options
        
    response: ChatResponse = ollama.chat(model, messages, **kwargs)
    output = { 
        'role': response.message.role,
        'content': response.message.content
    }
    
    if response.message.thinking: output['thinking'] = response.message.thinking

    return output




def stream_answer(messages: list[dict[str, str]], model: str, options: dict[str, Any] | None):
    kwargs = dict() 
    #print("Controllers stream answer")
    if options is not None:
        kwargs['options'] = options
    # parameter to ollama for set stream
    kwargs['stream'] = True
        
    for token in ollama.chat(model, messages=messages, **kwargs):
        yield token['message']['content']
            
    





def json_answer(messages: list[dict[str, str]], model: str, format: type[BaseModel], options: dict[str, Any] | None) -> BaseModel:
    kwargs = dict() 
    if options is not None:
        kwargs['options'] = options

    response: ChatResponse = ollama.chat(
        messages= messages,
        model=model,
        format=format.model_json_schema(),
        **kwargs
    )

    if response.message.content is None:
        raise ValueError("Error when generating structure output message")

    #print(messages)


    #print(response.message.content)

    return format.model_validate_json(response.message.content)





def get_embedding(text: str, model: str, local=True) -> np.ndarray:
    if local: 
        client = Client(
            host = os.environ.get('EMB_MODEL_HOST', "localhost:11434"),
        )
        result = client.embed(model, text)
    else:
        result = ollama.embed(model, text)
    return np.array(result['embeddings'][0])




def tool_calling(messages: list[dict[str, str]], tools: dict[str, Callable], model: str, options: dict[str, Any] | None) -> list[dict[str, str]]:
    kwargs = dict() 
    if options is not None:
        kwargs['options'] = options
    
    tools_func: list[Callable] = [tools[i] for i in tools]

    messages_start_length = len(messages)

    while True:
        response: ChatResponse = ollama.chat(
            messages= messages,
            model=model,
            tools = tools_func, 
            **kwargs
        )
        messages.append({
            'role': response.message.role,
            'content': response.message.content
        })
        if response.message.thinking: messages[-1]['thinking'] = response.message.thinking

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                if tc.function.name in tools:
                    result = tools[tc.function.name](**tc.function.arguments)
                    messages.append({
                        'role': 'tool', 
                        'tool_name': tc.function.name, 
                        'content': str(result)
                    })

        else: break


    return messages[messages_start_length:]




# --- return available models list ---
def available_models() -> list[str]:
    return [
        m['model'] for m in ollama.list()['models']
    ] 
# ------------------------------------




if __name__ == "__main__":
    from rich.console import Console
    from rich.pretty import pprint

    models = available_models()
    pprint(models)








#    def add(a: int, b: int) -> int:
#      """Add two numbers"""
#      """
#      Args:
#        a: The first number
#        b: The second number
#
#      Returns:
#        The sum of the two numbers
#      """
#      return a + b
#
#
#    def multiply(a: int, b: int) -> int:
#      """Multiply two numbers"""
#      """
#      Args:
#        a: The first number
#        b: The second number
#
#      Returns:
#        The product of the two numbers
#      """
#      return a * b
#
#
#
#    available_functions = {
#      'add': add,
#      'multiply': multiply,
#    }
#
#
#    messages = [{'role': 'user', 'content': 'What is (11434+12341)*412?'}]
#    tool_result = tool_calling(messages, tools=available_functions, model="gpt-oss:20b-cloud", options=None)
#    print(tool_result)
#
#    class Result(BaseModel):
#        result: int = Field(..., description="Result for user query.")
#
#
#    print(Result.model_json_schema())
#    prompt = f"""
#    Answer: {messages[-1]['content']}
#
#    Ensure the output conforms strictly to the JSON format provided.
#    If a field is not present, omit it (do not return null).
#    Only return the JSON, do not include any markdown or other text.
#
#    Json Schema:
#    {Result.model_json_schema()}
#    """
#
#    print(json_answer([{'role': 'user', 'content': prompt}], format=Result, model="gpt-oss:20b-cloud", options=None))
#
#
