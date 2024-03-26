FROM python:3.11

# RUN useradd -m -u 1000 user
RUN useradd -ms /bin/bash appuser

RUN apt-get update -y \
    && apt-get install -y make curl \
    && curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/appuser/.local python3 - \
    # && curl -sSL https://install.python-poetry.org | python3 - \
    # && curl -sSL https://install.python-poetry.org | python3 -  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/appuser/.local/bin:$PATH"

RUN poetry config virtualenvs.create false

COPY . /home/appuser/app

ENV HOME=/home/appuser
WORKDIR $HOME/app
RUN poetry install

# COPY --chown=appuser . $HOME/app
RUN chown -R appuser:appuser $HOME/app
USER appuser

CMD ["poetry", "run", "chainlit", "run", "app.py", "--port", "7860"]
