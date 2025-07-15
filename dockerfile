FROM python:3.10-slim


ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


WORKDIR /app


COPY App/ /app/
COPY Saved_Models/ /app/Saved_Models/


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000


HEALTHCHECK CMD curl --fail http://localhost:5000 || exit 1

# Start the app
CMD ["python", "app.py"]
