from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.agent import handle_user_input,SYSTEM_PROMPT


#from agent import run_agent_once  # ← import your logic

from fastapi.middleware.cors import CORSMiddleware

#from agent import handle_user_input, SYSTEM_PROMPT
from langchain_core.messages import SystemMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str



@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT)]
        }
    state = sessions[req.session_id]
    response, updated_state = handle_user_input(req.message, state)
    sessions[req.session_id] = updated_state
    return {"response": response}
