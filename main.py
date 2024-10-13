from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
import os
import pandas as pd

# подгружаем функции из других файлов
from codes.preprocessing import PreprocessUseCases
from codes.models import MyModel

app = FastAPI()

# Подключаем шаблоны HTML из директории
templates = Jinja2Templates(directory="codes/templates")
preprocess = PreprocessUseCases()
model = MyModel()
task_status = {}  # Словарь для хранения статусов задач


# главная страница
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("interactive.html", {"request": request})


# Настройка логирования
logging.basicConfig(level=logging.INFO)


# Функция для обработки текста пользователя или загруженных файлов
def get_usecase_path_list(user_text=None, uploaded_files=None):
    """Конвертирует введенный текст в .txt или обрабатывает загруженные файлы."""
    path_list = []

    if user_text:
        # Если введен текст, создаем .txt и добавляем в список
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

    # Возвращаем список файлов для обработки
    if path_list:
        return path_list


# Фоновая задача для запуска модели
def run_model_task(task_id, user_text=None, uploaded_files=None):
    global task_status
    path_list = get_usecase_path_list(
        user_text=user_text, uploaded_files=uploaded_files
    )

    if user_text:
        # Если был введен текст, результат - текст модели
        # file_contents = preprocess.get_summarized_data(path_list)
        # result_file_path = model.process_text(file_contents)
        result_file_path = "results/model_data.xlsx"

        # Читаем таблицу, извлекаем нужные данные
        df = pd.read_excel(result_file_path)
        usecase_text = df.iloc[0, 1]  # Текст юзкейса (колонка 1)
        certifiable_object = df.iloc[0, 2]  # Сертифицируемый объект (колонка 2)
        regulation_summary = df.iloc[0, 3]  # Выжимка из регламента (колонка 3)
        model_response = df.iloc[0, 4]  # Ответ модели (колонка 4)

        bytesio_file = ...
        # Сохраняем данные в словарь для последующей выдачи
        task_status[task_id] = {
            "type": "text",
            "result": {
                "usecase_text": usecase_text,
                "certifiable_object": certifiable_object,
                "regulation_summary": regulation_summary,
                "model_response": model_response,
                "download_file": bytesio_file.getvalue(),
            },
        }

    elif uploaded_files:
        # Если были загружены файлы, результат - Excel файл
        file_contents = preprocess.get_summarized_data(path_list)
        result_file_path = model.process_files(file_contents)
        df = pd.read_excel(result_file_path)
        bytesio_file = ...
        task_status[task_id] = {"type": "file", "result": bytesio_file.getvalue()}


# Обработка данных формы (файлы или текст)
@app.post("/submit")
async def submit_form(
    request: Request,
    background_tasks: BackgroundTasks,
    user_text: str = Form(None),
    files: list[UploadFile] = File(None),
):
    logging.info(f"Получен POST-запрос с user_text: '{user_text}', files: {files}")

    # Присваиваем ID задачи для отслеживания статуса
    task_id = "task_1"
    task_status[task_id] = None

    # Запускаем фоновую задачу для запуска модели
    background_tasks.add_task(run_model_task, task_id, user_text, files)

    # Отображаем страницу ожидания с неактивной кнопкой
    return templates.TemplateResponse(
        "waiting.html", {"request": request, "task_id": task_id}
    )


# Эндпоинт для проверки статуса задачи
@app.get("/check_status/{task_id}")
async def check_status(task_id: str):
    if task_status.get(task_id):
        return {"status": "completed"}  # Статус "выполнено"
    return {"status": "pending"}  # Статус "в ожидании"


# Страница с результатами
@app.get("/results/{task_id}", response_class=HTMLResponse)
async def results(request: Request, task_id: str):
    task_result = task_status.get(task_id)
    if task_result:
        if task_result["type"] == "text":
            # Рендерим страницу с текстовым результатом и кнопкой скачивания
            return templates.TemplateResponse(
                "result_text.html",
                {
                    "request": request,
                    "usecase_text": task_result["result"]["usecase_text"],
                    "certifiable_object": task_result["result"]["certifiable_object"],
                    "regulation_summary": task_result["result"]["regulation_summary"],
                    "model_response": task_result["result"]["model_response"],
                    "download_link": f"/download/temp_result.xlsx",
                },
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


# Эндпоинт для скачивания файла
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"results/{filename}"
    return HTMLResponse(
        content=open(file_path, "rb").read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    os.makedirs(
        "results", exist_ok=True
    )  # Создаем директорию для результатов, если она не существует
    uvicorn.run("main:app", host="0.0.0.0", port=8000)  # Запуск приложения FastAPI
