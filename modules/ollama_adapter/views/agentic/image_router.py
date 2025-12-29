from ...views import ollama as provider
from ...models.Answer import Answer, JSONFormat
from ...models.ollama import OllamaOptions
from pydantic import BaseModel

class ImageRoute(BaseModel):
    route: int

# --- router for image answer ---
def image_router(query: Answer, model="qwen3-vl:30b") -> int:

    # --- create prompt for route agent ---
    def return_prompt(user_request: str) -> str:
        return "\n".join([
            "# Role",
            " ".join([
                "You are an AI assistant, that decide",
                "what route should choose for process user request"
            ]),
            "",
            "# Instructions",
            " ".join([
                "1. Choose route *1*, if user specify",
                "that we need **only** generate bounding boxing for image",
            ]),
            " ".join([
                "2. Choose route *2*, if create bounding boxes to addition",
                "for answer will be usefull, maybe user search something on image."
            ]),
            " ".join([
                "3. Choose route *0*, if another conditions",
                "does not fit for user request."
            ]),
            "",
            "",
            f"User request: {user_request}"
        ])
    # -------------------------------------

    route: int = provider.json_output(
        query = JSONFormat(
            answer = Answer(
                query = return_prompt(query.query)
            ),
            format = ImageRoute
        ),
        model = model,
        options = OllamaOptions(temperature=0)
    ).output.route

    return route

# -------------------------------



if __name__ == "__main__":
    
    from rich.console import Console
    from rich.pretty import pprint

    console = Console()

    while True:
        request = console.input("[cyan]Type user request: [/]")
        with console.status("Generating route"):
            route: int = image_router(
                query = Answer(query=request)
            )
        console.print(f"[green]Route: {route}[/]")
        

    



