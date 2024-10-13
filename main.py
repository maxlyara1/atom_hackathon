from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
import os
from codes.preprocessing import PreprocessUseCases

# Import the model from the separate file
from codes.models import MyModel

app = FastAPI()

templates = Jinja2Templates(directory="codes/templates")
preprocess = PreprocessUseCases()
model = MyModel()
task_status = {}


# Main page with form
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("interactive.html", {"request": request})


# Настройка логирования
logging.basicConfig(level=logging.INFO)


def get_usecase_path_list(user_text=None, uploaded_files=None):
    """Convert input text to .txt or handle uploaded files, then preprocess."""
    path_list = []

    if user_text:
        # Если введен текст, создаем .txt файл
        txt_file_path = "uploads/input_text.txt"
        with open(txt_file_path, "w") as txt_file:
            txt_file.write(user_text)
        path_list.append(txt_file_path)

    elif uploaded_files:
        # Если загружены файлы, добавляем их в список
        for file in uploaded_files:
            file_location = f"uploads/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(file.file.read())
            path_list.append(file_location)
    # Если есть файлы для обработки
    if path_list:
        return path_list


# Background task to process the model
def run_model_task(task_id, user_text=None, uploaded_files=None):
    global task_status
    path_list = get_usecase_path_list(
        user_text=user_text, uploaded_files=uploaded_files
    )

    if user_text:
        # Если был введен текст, результат - текст модели
        file_contents = preprocess.get_summarized_data(path_list)
        response_text = model.process_text(file_contents)
        task_status[task_id] = {"type": "text", "result": response_text}
    elif uploaded_files:
        # Если были загружены файлы, результат - Excel файл
        file_contents = preprocess.get_summarized_data(path_list)
        result_file = model.process_files(file_contents)
        task_status[task_id] = {"type": "file", "result": result_file.getvalue()}


# Process form data (file or text)
@app.post("/submit")
async def submit_form(
    request: Request,
    background_tasks: BackgroundTasks,
    user_text: str = Form(None),
    files: list[UploadFile] = File(None),
):
    logging.info(f"Received POST request with user_text: '{user_text}', files: {files}")

    # Assign a task ID to track the status
    task_id = "task_1"
    task_status[task_id] = None

    # Start background task to run the model
    background_tasks.add_task(run_model_task, task_id, user_text, files)

    # Render the waiting page with inactive button
    return templates.TemplateResponse(
        "waiting.html", {"request": request, "task_id": task_id}
    )


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
            # Рендерим страницу с текстовым результатом
            return templates.TemplateResponse(
                "result_text.html",
                {"request": request, "response_text": task_result["result"]},
            )
        elif task_result["type"] == "file":
            # Сохраняем Excel файл и рендерим страницу для его скачивания
            temp_file_path = "results/temp_result.xlsx"
            with open(temp_file_path, "wb") as f:
                f.write(task_result["result"])
            return templates.TemplateResponse(
                "result_file.html",
                {
                    "request": request,
                    "result": f"/download/{os.path.basename(temp_file_path)}",
                },
            )
    return RedirectResponse("/")


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"results/{filename}"
    return HTMLResponse(
        content=open(file_path, "rb").read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
