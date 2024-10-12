from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import uvicorn
import logging
import io
import os
import time  # Для симуляции долгой работы

# Import the model from the separate file
from models.models import MyModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = MyModel()
task_status = {}

# Main page with form
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("interactive.html", {"request": request})

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Background task to process the model
def run_model_task(task_id, user_text=None, files=None):
    global task_status
    if user_text:
        response_text = model.process_text(user_text)
        task_status[task_id] = {"type": "text", "result": response_text}
    elif files:
        file_contents = []
        for file in files:
            file_contents.append(file)
        result_file = model.process_files(file_contents)
        # Assuming result_file has a way to be converted to a file path or binary content
        task_status[task_id] = {"type": "file", "result": result_file.getvalue()}

# Process form data (file or text)
@app.post("/submit")
async def submit_form(request: Request, background_tasks: BackgroundTasks, user_text: str = Form(None), files: list[UploadFile] = File(None)):
    logging.info(f"Received POST request with user_text: '{user_text}', files: {files}")

    # Assign a task ID to track the status
    task_id = "task_1"
    task_status[task_id] = None

    # Start background task to run the model
    background_tasks.add_task(run_model_task, task_id, user_text, files)

    # Render the waiting page with inactive button
    return templates.TemplateResponse("waiting.html", {"request": request, "task_id": task_id})

# Endpoint to check task status
@app.get("/check_status/{task_id}")
async def check_status(task_id: str):
    if task_status.get(task_id):
        return {"status": "completed"}
    return {"status": "pending"}

# Results page
@app.get("/results/{task_id}", response_class=HTMLResponse)
async def results(request: Request, task_id: str):
    task_result = task_status.get(task_id)
    if task_result:
        if task_result["type"] == "text":
            return templates.TemplateResponse("result_text.html", {"request": request, "response_text": task_result["result"]})
        elif task_result["type"] == "file":
            # Assuming result_file is saved or available to serve
            temp_file_path = "results/temp_result.xlsx"
            # Render the result file page with the download link
            return templates.TemplateResponse("result_file.html", {"request": request, "result": f"/download/{os.path.basename(temp_file_path)}"})
    return RedirectResponse("/")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"results/{filename}"
    return HTMLResponse(content=open(file_path, "rb").read(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename={filename}"})

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
