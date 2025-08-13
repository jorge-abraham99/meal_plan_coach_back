import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
from fastapi.responses import JSONResponse

from prompts import system_prompt
from call_function import available_functions, call_function
from mangum import Mangum


load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",  # update with your Next.js URL if different
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_ITERS = 85

# Initialize the Gemini client once
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY not set in environment")
client = genai.Client(api_key=api_key)


def generate_agent_response(user_prompt: str, verbose: bool = False) -> str:
    messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]
    iters = 0

    while True:
        iters += 1
        if iters > MAX_ITERS:
            raise Exception(f"Maximum iterations ({MAX_ITERS}) reached.")

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                tools=[available_functions], system_instruction=system_prompt
            ),
        )

        if verbose:
            print("Prompt tokens:", response.usage_metadata.prompt_token_count)
            print("Response tokens:", response.usage_metadata.candidates_token_count)

        if response.candidates:
            for candidate in response.candidates:
                function_call_content = candidate.content
                messages.append(function_call_content)

        if not response.function_calls:
            return response.text

        function_responses = []
        for function_call_part in response.function_calls:
            function_call_result = call_function(function_call_part, verbose)
            if (
                not function_call_result.parts
                or not function_call_result.parts[0].function_response
            ):
                raise Exception("empty function call result")
            if verbose:
                print(f"-> {function_call_result.parts[0].function_response.response}")
            function_responses.append(function_call_result.parts[0])
        if not function_responses:
            raise Exception("no function responses generated, exiting.")

        messages.append(types.Content(role="tool", parts=function_responses))


@app.get("/chat", response_class=PlainTextResponse)
def generate(prompt: str = Query(..., description="User prompt to generate the agent response"), verbose: bool = False):
    try:
        final_response = generate_agent_response(prompt, verbose)
        return final_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_meal_plan", response_class=JSONResponse)
def get_meal_plan():
    file_path = os.path.join(os.path.dirname(__file__), "meal_plan.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="meal_plan.json not found")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
