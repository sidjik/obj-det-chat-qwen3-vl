from ...views import ollama as provider
from ...models.Answer import ImageAnswer, JSONFormat
from ...models.ollama import OllamaOptions
from pydantic import BaseModel
from pathlib import Path
import numpy as np



class ObjEntity(BaseModel):
    coordinates: tuple[int, int, int, int]
    label: str

# --- generate coordinates for object ---
def generate_coordinates(query: ImageAnswer, model: str = "qwen3-vl:30b") -> list[ObjEntity]:

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
            "",
        ])
    }]
    
    coordinates: list[ObjEntity] = provider.json_output(
        query =  JSONFormat(
            answer = query,
            format = ListPlug
        ),
        model = model,
        options = OllamaOptions(
            top_p = 0.9,
            top_k = 30,
            temperature = 0
        )
    ).output.result
    
    return coordinates
# ---------------------------------------



if __name__ == "__main__":

    from rich.console import Console
    from rich.pretty import pprint

    console = Console()

    #with open(Path("~/Desktop/gotou2.jpg").expanduser(), 'rb') as f: 
    with open(Path("~/Desktop/football_field.jpg").expanduser(), 'rb') as f: 
        img_b = f.read()

    answer = ImageAnswer(
        #query = "Provide three bounding boxes of anime character head and both arms in JSON format.",
        query = "Provide bounding boxes for all football player in red t-shirts.",
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



