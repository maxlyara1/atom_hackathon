from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import BytesIO
import uvicorn
import os

app = FastAPI()

# Указываем путь к папке с шаблонами
templates = Jinja2Templates(directory="templates")

# Папки для загрузки файлов и результатов
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Рендеринг HTML страницы
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("interactive.html", {"request": request})

# Обработчик загрузки файлов
@app.post("/upload")
async def upload_files(name: str = Form(...), files: list[UploadFile] = File(...)):
    file_paths = []

    # Сохраняем загруженные файлы
    for file in files:
        file_location = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        file_paths.append(file_location)

    # Обработка файлов через модель (тут вызывается функция, обрабатывающая данные)
    result_file_path = process_files(file_paths, name)

    # Возвращаем HTML с ссылкой на скачивание результата
    return templates.TemplateResponse("interactive.html", {"request": Request, "result": f"/download/{os.path.basename(result_file_path)}"})

# Функция для обработки файлов (здесь можно добавить логику модели)
def process_files(file_paths, requirement):
    all_data = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)  # Пример чтения Excel
        df['Требование'] = requirement  # Добавляем новое требование
        all_data.append(df)

    # Объединяем данные и сохраняем в Excel
    result_df = pd.concat(all_data, ignore_index=True)
    result_filename = f"{RESULT_DIR}/result_{requirement}.xlsx"
    result_df.to_excel(result_filename, index=False)

    return result_filename

# Endpoint для скачивания файла
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"{RESULT_DIR}/{filename}"
    return FileResponse(path=file_path, filename=filename)

# # Статические файлы
# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
