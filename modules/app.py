import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider, Tags, TextInput
import os 
import requests
import codecs
from pathlib import Path
import requests
import base64
import codecs
import supervision as sv
import json
from ollama_adapter.views.tools.bounding_box import * 
from PIL import Image
import io
import logging
from rich.logging import RichHandler
from ollama_adapter.models.ollama import OllamaOptions
import pprint

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)



log = logging.getLogger("rich")

OLLAMA_ADAPTER_HOST: str = os.environ.get("OLLAMA_ADAPTER_HOST", "http://localhost:8000")


DEFAULT_OLLAMA_OPTIONS = OllamaOptions(
    typical_p=0.7,
    top_k=9, 
    mirostat=2,
    repeat_penalty=0.3,
    repeat_last_n=-1
)



@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None





async def get_async_response(message): 
    for i in message: 
        yield i


@cl.on_message
async def main(message: cl.Message):
    msg: cl.Message = cl.Message(content="")
    log.info(message)
    models_info: list[dict] | None = cl.user_session.get('models_info')

    if models_info is None: raise ValueError('Cannot rich models info.')

    model_name = cl.user_session.get('model')

    vision_model: bool = any([ _['name'] == model_name and _['vision_model'] for _ in models_info])
    elements: list[cl.Image] = []

    history: list = cl.user_session.get('history') or list()
    history += [{'role': 'user', 'content': message.content}]
    history += [{'role': 'assistant', 'content': ''}]
    log.info(pprint.pformat(history))

    settings: dict[str, str] = cl.user_session.get('settings') or {
        'history_limit': '3',
        'num_ctx': '4096',
        'num_keep': '4096',
        'seed': '42',
        'temperature': '0.0'
    }
    log.info(pprint.pformat(settings))
    history_limit: int = int(settings.pop('history_limit', 3))
    try:

        if len(message.elements): 
            # --- image answer ---
            if vision_model:
                images_bytes: list[bytes] = []
                images_bytes_decode: list = []
                for element in message.elements:
                    with open(element.path, "rb") as f:
                        img_b = f.read()
                        b64 = base64.b64encode(img_b).decode("ascii")
                        images_bytes += [img_b]
                        images_bytes_decode += [b64]



                with requests.post(
                    url=f"{OLLAMA_ADAPTER_HOST}/ollama/image/obj-det-pipeline", 
                    json={
                        'query': {
                            'query': message.content,
                            'paths': images_bytes_decode,
                            'other_dict': history[-(history_limit+2):-2]
                        },
                        'options': {**DEFAULT_OLLAMA_OPTIONS.get_dict, **settings} 
                    },
                    stream=True
                ) as r:

                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line: continue

                        data = json.loads(line)

                        if data["type"] == "define-route":
                            async with cl.Step(name="Image Routing") as step:
                                step.output = f"Route: {data['data']}\n"
                                if data['data'] == 1:
                                    step.output += "\n".join([
                                        "Agent provide to you only image with bounding box that define."
                                    ])
                                elif data['data'] == 2:
                                    step.output += "\n".join([
                                        "Agent at first search entity on image and we search his on image.",
                                        "After that final output will be generated.",
                                        "Also for user agent provide picture with bounding object."
                                    ])
                                else:
                                    step.output += "\n".join([
                                        "Agent process image and answer on user question."
                                    ])


                        elif data["type"] == "search-entity":
                            async with cl.Step(name="Find entity") as step:
                                step.output = f"Generated search entity: {data['data']}\n" + "\n".join([
                                    "This entity will be provided for agent that search entity on image"
                                ])

                        
                        elif data['type'] == 'coordinates':
                            coor = data["data"]
                            async with cl.Step(name="Bounding objects coordinates") as step:
                                coordinates: list[tuple[int, int, int, int]] = [ ent['coordinates'] for ent in coor ]
                                labels: list[str] = [ ent['label'] for ent in coor ]
                                abs_coors = normalize_coordinates(coordinates, Image.open(io.BytesIO(images_bytes[0])).size)
                                step.input = pprint.pformat(
                                    {'coor': abs_coors, 'labels': labels},
                                    width=100,
                                    sort_dicts=False
                                )

                            img: np.ndarray = annotate(images_bytes[0], abs_coors, labels)
                            elements.append(cl.Image(
                                name = "image1.jpg",
                                display = "inline",
                                content = cv2.imencode('.jpg', img)[-1].tobytes()
                            ))
                            

                        elif data["type"] == "text-token":
                            history[-1]['content'] += str(data['data'])
                            await msg.stream_token(str(data['data']))



            else: 
                # --- GENERATE RESPONSE THAT MODEL NOT SUPPORT IMAGE ---
                with requests.post(
                    url=f"{OLLAMA_ADAPTER_HOST}/ollama/text/answer/stream?model={model_name}", 
                    json={
                        'query': {
                            'query': "Simply generate response that model(you) does not support image capatibilites.",
                            'role': "system"
                        },
                        'opt': {**DEFAULT_OLLAMA_OPTIONS.get_dict} 
                    },
                    stream=True
                ) as token_stream:

                    decoder = codecs.getincrementaldecoder("utf-8")()

                    for chunk in token_stream.iter_content(chunk_size=3):
                        if not chunk:
                            continue

                        text = decoder.decode(chunk)
                        if text:
                            history[-1]['content'] += text
                            await msg.stream_token(text)

                    tail = decoder.decode(b"", final=True)
                    if tail:
                        history[-1]['content'] += tail
                        await msg.stream_token(tail)
                # ------------------------------------------------------


            
            # --------------------
        else: 
            
            # --- ANSWER FOR TEXT REQUEST --- 
            with requests.post(
                url=f"{OLLAMA_ADAPTER_HOST}/ollama/text/answer/stream?model={model_name}", 
                json={
                    'query': {
                        'query': message.content,
                        'other_dict': history[-(history_limit+2):-2]
                    },
                    'opt': {**DEFAULT_OLLAMA_OPTIONS.get_dict, **settings} 
                },
                stream=True
            ) as token_stream:

                decoder = codecs.getincrementaldecoder("utf-8")()

                for chunk in token_stream.iter_content(chunk_size=3):
                    if not chunk:
                        continue

                    text = decoder.decode(chunk)
                    if text:
                        history[-1]['content'] += text
                        await msg.stream_token(text)

                tail = decoder.decode(b"", final=True)
                if tail:
                    history[-1]['content'] += tail
                    await msg.stream_token(tail)
            # ------------------------------- 
        
        if msg.content != "": 
            await msg.send()
        else:
            history = history[:-2]
        cl.user_session.set('history', history)
        # --- send images (if we have) ---
        if len(elements): await cl.Message(
            content="This message has an image!",
            elements=elements,
        ).send()
        # --------------------------------
    except Exception as e: 
        msg = cl.Message(content='Something went wrong: {}'.format(e))
        await msg.send()
        #raise e




from chainlit.types import ThreadDict

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):

    model_name = thread['metadata']['chat_profile']

    history: list[dict[str, str]] = []
    for step in thread['steps']:
        history.append({
            'role': step['name'].lower() if step['name'] == 'assistant' else 'user',
            'content': step['output']
        })
    cl.user_session.set('history', history)
        

    cl.user_session.set('model', model_name)







@cl.on_chat_start
async def start():
    model_name = cl.user_session.get("chat_profile")

    # --- POST SETTINGS TO CHAINLIT ---
    await cl.ChatSettings([
        TextInput(
            id="num_ctx", 
            label="Context window size", 
            initial="4096",       
            description='Sets the size of the context window for generating the next token (the size of the history the model works with).'
        ),
        TextInput(
            id="num_keep", 
            label="Keep token size", 
            initial="4096", 
            description='Specifies the number of tokens to preserve when generating text (e.g. to preserve a certain part of the context).'
        ),
        Slider(
            id='temperature', 
            label='Set temperature number',
            initial = 0,
            min = 0,
            max=1,
            step=0.01,
        ),
        TextInput(
            id='seed', 
            label='Seed number', 
            initial='42', 
            description='Sets the seed for random number generation. Allows predictions to be reproducible.'
        ),
        Slider(
            id = 'history_limit',
            label = 'Set history limit message', 
            initial = 3, 
            min = 0, 
            max = 12, 
            step = 1,
        )
    ]).send()
    # ---------------------------------
    # --- SETUP INITIAL VOCAB WITH SETTINGS --- 
    cl.user_session.set('settings', {
        'history_limit': '3',
        'num_ctx': '4096',
        'num_keep': '4096',
        'seed': '42',
        'temperature': '0.0'
    })
    # ----------------------------------------- 

    cl.user_session.set('model', model_name)
    
    # --- GET MODELS INFO ---
    models = requests.get(
        url = f"{OLLAMA_ADAPTER_HOST}/ollama/available_models"
    ).json()
    cl.user_session.set('models_info', models)
    # -----------------------

    cl.user_session.set('history', [])
    









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
    return [
        cl.Starter(
            label="Quantum entanglement, intuitively",
            message="Explain quantum entanglement in a way that remains physically accurate while using intuitive analogies. Then explain where those analogies break down.",
        ),

        cl.Starter(
            label="How modern semiconductors work",
            message="Explain how modern semiconductors work in detail: p–n junctions, band theory, doping, and why shrinking process nodes makes engineers' lives harder.",
        ),

        cl.Starter(
            label="Training a neural network from scratch",
            message="Describe the process of training a neural network from scratch: from the mathematical model of a neuron to backpropagation and gradient descent. No code, but rigorous and structured.",
        ),

        cl.Starter(
            label="Why chaos can be deterministic",
            message="Explain how deterministic systems can exhibit chaotic behavior. Use examples such as the logistic map or the double pendulum.",
        )
    ]




