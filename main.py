from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import uvicorn
import logging
import io

# Import the model from the separate file
from models.models import MyModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = MyModel()

# Main page with form
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("interactive.html", {"request": request})

# Настройка логирования
logging.basicConfig(level=logging.INFO)

@app.post("/submit")
async def submit_form(request: Request, files: list[UploadFile] = File(None), user_text: str = Form(None)):
    logging.info(f"Received POST request with files: {files}, user_text: '{user_text}'")

    if user_text:
        # If user submitted text
        response_text = f"Модель ответила: '{model.process_text(user_text)}'"
        return templates.TemplateResponse("result_text.html", {"request": request, "response_text": response_text})
    
    elif files:
        # If files were uploaded
        file_contents = []
        for file in files:
            # Read the uploaded file into memory
            content = await file.read()
            file_contents.append(content)

        # Process the uploaded files to generate an empty Excel file in memory
        result_file = model.process_files(file_contents)

        # Create a response to download the resulting file
        return HTMLResponse(content=result_file.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=result.xlsx"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
