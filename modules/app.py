import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider, Tags, TextInput
from OllamaModel import *
import os 
import ollama
import requests
import codecs


OLLAMA_ADAPTER_HOST: str = os.environ.get("OLLAMA_ADAPTER_HOST", "http://localhost:8000")




@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None



vision_model = ['llava', 'moondream', 'llama3.2-vision', 
                'llava-llama3', 'bakllava', 'minicpm-v']


async def get_async_response(message): 
    for i in message: 
        yield i


@cl.on_message
async def main(message):
    msg = cl.Message(content="")
    try:
        model_name = cl.user_session.get('model')

        if len(message.elements): 
            # --- image answer ---
            if any(i in model.name for i in vision_model):
               
                # i don`t know how it works :), because i can`t run vision model localy on my pc :(
                paths = list()
                for element in message.elements: 
                    os.system('docker cp {} /.'.format(element.path))
                    paths.append('/{}'.format(element.path))
                answer = model.get_response_with_image(message.content, paths)
            else: 
                answer = get_async_response('This model does not support image to text compatibility')
            # --------------------
        else: 
            
            with requests.post(
                url=f"{OLLAMA_ADAPTER_HOST}/ollama/text/answer/stream?model={model_name}", 
                json={
                    'query': {
                        'query': message.content
                    }
                },
                stream=True
            ) as token_stream:

                decoder = codecs.getincrementaldecoder("utf-8")()

                for chunk in token_stream.iter_content(chunk_size=3):
                    if not chunk:
                        continue

                    text = decoder.decode(chunk)
                    if text:
                        await msg.stream_token(text)

                tail = decoder.decode(b"", final=True)
                if tail:
                    await msg.stream_token(tail)

            #async for token in token_stream:
                #await msg.stream_token(str(token))

        await msg.send()

    except Exception as e: 
        msg = cl.Message(content='Something went wrong: {}'.format(e))
        await msg.send()
        raise e




from chainlit.types import ThreadDict

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    #for step in thread['steps']:
    #    print(step['name'], step['output'])
    #    if step['name'].lower() == 'assistant':
    #        name = 'assistant'
    #    else: 
    #        name = 'user'
    #    model.append_message(name, step['output'])

    model_name = thread['metadata']['chat_profile']
    cl.user_session.set('model', model_name)







@cl.on_chat_start
async def start():
    model_name = cl.user_session.get("chat_profile")

    settings = await cl.ChatSettings(
        [
            TextInput(id="num_ctx", label="Context window size", initial="4096", 
                      description='Sets the size of the context window for generating the next token (the size of the history the model works with).'),
            TextInput(id="num_keep", label="Keep token size", initial="4096", 
                      sdescription='Specifies the number of tokens to preserve when generating text (e.g. to preserve a certain part of the context).'),
            Slider(
                id='temperature', 
                label='Set temperature number',
                initial = 0,
                min = 0,
                max=1,
                step=0.01,
            ),
            TextInput(id='seed', label='Seed number', initial='21', 
                      description='Sets the seed for random number generation. Allows predictions to be reproducible.'),
            Slider(
                id = 'history_limit',
                label = 'Set history limit message', 
                initial = 3, 
                min = 0, 
                max = 12, 
                step = 1,
            )

        ]
        ).send()

    cl.user_session.set('model', model_name)
    
    









@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set('settings', settings)






@cl.set_chat_profiles
async def chat_profile():
    chainlit_list: list[cl.ChatProfile] = []
    # --- get available models ---
    models = requests.get(
        url = f"{OLLAMA_ADAPTER_HOST}/ollama/available_models"
    ).json()
    for model in models:
        chainlit_list.append(cl.ChatProfile(
            name = model['name'],
            markdown_description = model['desc'],
            icon='https://images.seeklogo.com/logo-png/59/1/ollama-logo-png_seeklogo-593420.png'
        ))
    # ----------------------------
    return chainlit_list 


    




@cl.set_starters
async def set_starters():
    # !!! change this shit
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            )
        ]




