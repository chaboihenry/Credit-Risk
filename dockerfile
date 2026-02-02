FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Default Command: opens a bash terminal so you can type commands manually
CMD ["/bin/bash"]


