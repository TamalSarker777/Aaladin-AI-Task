# from fastapi import FastAPI
# from pydantic import BaseModel
# from .function import chat_infer  # <- Add the dot here

# app = FastAPI(
#     title="DeepSeek LoRA Inference API ",
#     description="Chat with your fine-tuned DeepSeek LoRA model right here in Swagger UI!",
#     version="1.0.0",
# )


# class Query(BaseModel):
#     prompt: str = "Hello, how are you?"
#     max_new_tokens: int = 128


# @app.get("/")
# async def root():
#     return {"message": "DeepSeek LoRA API is running "}


# @app.post("/infer", summary="Chat with the fine-tuned model", tags=["Inference"])
# async def infer(data: Query):
#     """
#     Send a prompt to the fine-tuned model and get the generated response.
#     """
#     result = chat_infer(data.prompt, data.max_new_tokens)
#     return {"response": result}



from fastapi import FastAPI
from pydantic import BaseModel
from .function import chat_infer

app = FastAPI(
    title="DeepSeek LoRA Inference API",
    description="Chat with your fine-tuned DeepSeek LoRA model right here in Swagger UI!",
    version="1.0.0",
)


class Query(BaseModel):
    prompt: str = "Hello, how are you?"
    max_new_tokens: int = 128


@app.get("/")
async def root():
    return {"message": "DeepSeek LoRA API is running"}


@app.post("/infer", summary="Chat with the fine-tuned model", tags=["Inference"])
async def infer(data: Query):
    """
    Send a prompt to the fine-tuned model and get the generated response.
    """
    result = chat_infer(data.prompt, data.max_new_tokens)
    return {"response": result}


# Run the server
# uvicorn main:app --host 0.0.0.0 --port 8000
