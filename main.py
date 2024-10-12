from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import uvicorn
import logging
import io
import os

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

# Process form data (file or text)
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

        # Save the Excel file to a temporary location
        temp_file_path = "results/temp_result.xlsx"
        with open(temp_file_path, "wb") as f:
            f.write(result_file.getvalue())

        # Render the result file page with the download link
        return templates.TemplateResponse("result_file.html", {"request": request, "result": f"/download/{os.path.basename(temp_file_path)}"})

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"results/{filename}"
    return HTMLResponse(content=open(file_path, "rb").read(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename={filename}"})

if __name__ == "__main__":
    # Make sure the 'results' directory exists
    os.makedirs("results", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
