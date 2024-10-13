from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
import os
import pandas as pd
from io import BytesIO

# подгружаем функции из других файлов
from codes.preprocessing import TextPreprocessor
from codes.preprocessing import PreprocessUseCases
from codes.preprocessing import PreprocessRegulations
from codes.preprocessing import GetPairs
from codes.models import MyModel

app = FastAPI()

# Подключаем шаблоны HTML из директории
templates = Jinja2Templates(directory="codes/templates")

text_preprocessor = TextPreprocessor()
preprocess_use_case = PreprocessUseCases()
preprocess_regulations = PreprocessRegulations()
model = MyModel()
get_pairs = GetPairs()

task_status = {}  # Словарь для хранения статусов задач


# Главная страница
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

    return path_list


# Фоновая задача для запуска модели
def run_model_task(task_id, user_text=None, uploaded_files=None):
    global task_status
    path_list = get_usecase_path_list(
        user_text=user_text, uploaded_files=uploaded_files
    )

    if user_text:
        # Подготовка данных через PreprocessUseCases
        # df_usecase = preprocess_use_case.get_summarized_data(path_list)
        # df_regulations = preprocess_regulations.get_embeddings_df()
        # # Получение пар текстов для модели
        # df_for_model = get_pairs.get_two_texts(df_usecase, df_regulations)
        # # Применение модели для обработки данных
        # result_df_path = model.model_work(df_for_model)
        # df_result = pd.read_excel(result_df_path)

        result_df_path = "results/model_data.xlsx"
        df_result = pd.read_excel(result_df_path)

        # Сохраняем результат в BytesIO для скачивания
        bytesio_file = BytesIO()
        df_result.to_excel(bytesio_file, index=False)
        bytesio_file.seek(0)

        # Сохраняем результат в словарь
        task_status[task_id] = {
            "type": "text",
            "result": {
                "usecase_text": df_result.iloc[0, 0],  # Текст юзкейса
                "certifiable_object": df_result.iloc[0, 1],  # Сертифицируемый объект
                "regulation_summary": df_result.iloc[0, 2],  # Выжимка из регламента
                "model_response": df_result.iloc[0, 3],  # Ответ модели
                "download_file": bytesio_file,
            },
        }

    elif uploaded_files:
        # # Подготовка данных через PreprocessUseCases
        # df_usecase = preprocess_use_case.get_summarized_data(path_list)
        # df_regulations = preprocess_regulations.get_embeddings_df()
        # # Получение пар текстов для модели
        # df_for_model = get_pairs.get_two_texts(df_usecase, df_regulations)
        # # Применение модели для обработки данных
        # result_df_path = model.model_work(df_for_model)
        # df_result = pd.read_excel(result_df_path)

        # result_df_path = "results/model_data.xlsx"
        # df_result = pd.read_excel(result_df_path)

        # Сохраняем в BytesIO
        bytesio_file = BytesIO()
        df_result.to_excel(bytesio_file, index=False)
        bytesio_file.seek(0)

        task_status[task_id] = {"type": "file", "result": bytesio_file}


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
                    "download_link": f"/download/{task_id}",  # Используем ID задачи
                },
            )
        elif task_result["type"] == "file":
            # Рендерим страницу для скачивания файла
            return templates.TemplateResponse(
                "result_file.html",
                {
                    "request": request,
                    "result": f"/download/{task_id}",  # Используем ID задачи
                },
            )
    return RedirectResponse("/")


# Эндпоинт для скачивания файла
@app.get("/download/{task_id}")
async def download_file(task_id: str):
    task_result = task_status.get(task_id)
    if task_result and task_result["result"]:
        # Передаем BytesIO как поток для скачивания
        return StreamingResponse(
            task_result["result"]["download_file"],
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={task_id}.xlsx"},
        )
    return {"error": "File not found"}


if __name__ == "__main__":
    os.makedirs(
        "results", exist_ok=True
    )  # Создаем директорию для результатов, если она не существует
    uvicorn.run("main:app", host="0.0.0.0", port=8000)  # Запуск приложения FastAPI
