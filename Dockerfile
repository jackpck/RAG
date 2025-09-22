FROM python:3.12.3

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["python", "-m", "streamlit" ,"run", "src/app.py"]