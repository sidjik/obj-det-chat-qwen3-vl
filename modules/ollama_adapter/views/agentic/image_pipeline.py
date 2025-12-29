from ...views import ollama as provider
from ...models.Answer import *
from ...models.ollama import OllamaOptions
from .image_router import image_router as router 
from ..tools import object_detection_agent as bb_agent








# --- pipeline for process image request ---
def image_pipeline(query: ImageAnswer, model: str = "qwen3-vl:30b"):
    pass
    # --- router --- 
    route: int = router(
        query = query,
        model = model
    )
    # -------------- 
    yield {"type": "define-route", "data": route}

    if route == 1:
        # --- bounding box generate ---
        coor = bb_agent.generate_coordinates(
            query=query
        )
        yield {"type": "coordinates", "data": coor}
        # -----------------------------
    elif route == 2:
        pass
        # --- generate with mark neccesary entity ---
        search_entity: str | None = provider.answer(
            query = ImageAnswer(**{
                **query.model_dump(),
                "query": "\n".join([
                    "# Role",
                    " ".join([
                        "You are an AI assistant, that generate request",
                        "for another model that make bounding object detection."
                    ]),
                    "",
                    "# Instructions",
                    "1. Generate **only** what need detect in image for user request.",
                    "2. Generate **only** list with entity for detection.",
                    "3. Do not provide extra info."
                    "",
                    f"User request: {query.query}"
                ]),
            }),
            model = model,
            options = OllamaOptions(temperature=0)
        ).answer 
        if search_entity is None:
            raise ValueError("Search Entity is None")

        yield {"type": "search-entity", "data": search_entity}
        
        coor = bb_agent.generate_coordinates(
            query=ImageAnswer(**{
                **query.model_dump(), "query": f"Search entity: {search_entity}"
            })
        )
        yield {"type": "coordinates", "data": coor}

        for token in provider.stream_answer(
            query = ImageAnswer(**{
                **query.model_dump(),
                "query": "\n".join([
                    "# Role",
                    " ".join([
                        "You are an AI assistant that generate answer for user request",
                        "based on detected object on user provided image."
                    ]),
                    "",
                    "",
                    f"User request: {query.query}",
                    f"Detected object: {search_entity}",
                    ""
                ])
            }),
            model = model,
            options = OllamaOptions(
                temperature=0
            )
        ):

            yield {"type": "text-token", "data": token}
        


        # -------------------------------------------
    else:
        # --- simple answer on question ---
        for token in provider.stream_answer(
            query = query,
            model = model,
            options = OllamaOptions(temperature=0)
        ):

            yield {"type": "text-token", "data": token}
        # ---------------------------------

    
# ------------------------------------------







if __name__ == "__main__":
    from rich.console import Console
    from rich.pretty import pprint

    console = Console()

    # --- read test image ---
    from pathlib import Path
    with open(Path("~/Desktop/fridge.jpg").expanduser(), "rb") as f:
        image_b: bytes = f.read()
    # -----------------------


    while True:
        request: str = console.input("[cyan]Type user request: [/]")
        answer: ImageAnswer = ImageAnswer(
            query = request,
            paths = [image_b]
        )

        for data in image_pipeline(
            query = answer,
            model = "qwen3-vl:30b",
        ):
            if data["type"] == "define-route" or data["type"] == "search-entity":
                pprint(data['data'])
            elif data['type'] == 'coordinates':
                coor = data["data"]
                pprint(coor)
                coordinates = [ ent.coordinates for ent in coor ]
                labels = [ ent.label for ent in coor ]
                from ..tools.bounding_box import * 
                from PIL import Image
                import io
                abs_coors = normalize_coordinates(coordinates, Image.open(io.BytesIO(image_b)).size)
                pprint(abs_coors)


                img = annotate(image_b, abs_coors, labels)
                sv.plot_image(img)
                console.print(f"Image type: {type(img)}")
            elif data["type"] == "text-token":
                console.print(f"[cyan]{data["data"]}[/]", end="")


        console.print()









