FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu11
RUN pip install accelerate==0.28.0
RUN pip install bitsandbytes
RUN pip install faiss-gpu
RUN pip install -r requirements.txt
RUN chmod 777 /app
EXPOSE 5003
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]