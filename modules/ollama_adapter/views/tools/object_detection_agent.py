from ...views import ollama as provider
from ...models.Answer import ImageAnswer, JSONFormat
from ...models.ollama import OllamaOptions
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import logging

log = logging.getLogger("rich")



class ObjEntity(BaseModel):
    coordinates: tuple[int, int, int, int]
    label: str

# --- generate coordinates for image request---
def generate_coordinates(query: ImageAnswer, model: str = "qwen3-vl:30b") -> list[ObjEntity]:

    log.info("Start generating coordinates")

    class ListPlug(BaseModel):
        result: list[ObjEntity]


    query.other_dict = [{
        'role': 'system',
        'content': "\n".join([
            "# Role",
            "You are an AI assistant that make object detection",
            "",
            "# Goal",
            "Your goal is as accurately as possible generate coordinates for user request.",
            "",
            "# Instructions",
            "1. Make labels name clear and simple.",
            "2. Identical labels names means the same class.",
            "3. You can provide different names for different entity."
            "",
        ])
    }]
    log.info(f"Query: {query.query}")
    
    coordinates: list[ObjEntity] = provider.json_output(
        query =  JSONFormat(
            answer = query,
            format = ListPlug
        ),
        model = model,
        options = OllamaOptions(
            seed=42,
            typical_p=0.9,
            top_k=3, 
            temperature=0.3,
            mirostat=2,
            repeat_penalty=0.3,
            repeat_last_n=-1
        ),
        #cloud=True
    ).output.result


    log.info("Finish generating coordinates")
    
    return coordinates
# ---------------------------------------



if __name__ == "__main__":

    from rich.console import Console
    from rich.pretty import pprint

    console = Console()

    #with open(Path("~/Desktop/gotou2.jpg").expanduser(), 'rb') as f: 
    #with open(Path("~/Desktop/football_field.jpg").expanduser(), 'rb') as f: 
    with open(Path("~/Desktop/suricatas.jpeg").expanduser(), 'rb') as f: 
        img_b = f.read()

    answer = ImageAnswer(
        #query = "Provide three bounding boxes of anime character head and both arms in JSON format.",
        #query = "Provide bounding boxes for all football player of each team and prvodie for them separate labels.",
        #query = "Provide bounding boxes for all suricates that you see on screen, make box only with head for each one.",
        query = "provide bouding boxes of suricatas heads",
        paths = [img_b]
    )
    with console.status("Generate coordinates..."):
        coor = generate_coordinates(query=answer)
        print(coor)
    
    coordinates = [ ent.coordinates for ent in coor ]
    labels = [ ent.label for ent in coor ]

    
    from .bounding_box import * 
    from PIL import Image
    import io
    abs_coors = normalize_coordinates(coordinates, Image.open(io.BytesIO(img_b)).size)
    pprint(abs_coors)


    img = annotate(img_b, abs_coors, labels)
    sv.plot_image(img)
    console.print(f"Image type: {type(img)}")



