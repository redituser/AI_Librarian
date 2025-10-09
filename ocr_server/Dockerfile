FROM python:3.10-slim

WORKDIR /app

RUN pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install "paddleocr>=2.0.1"
RUN pip install fastapi uvicorn python-multipart Pillow numpy opencv-python-headless

COPY . .

CMD ["uvicorn", "main:app" , "--host" , "0.0.0.0" , "--port" , "8000"]




