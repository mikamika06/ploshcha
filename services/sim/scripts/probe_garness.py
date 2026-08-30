import re
import subprocess
import time

REPO = "/Users/macbook/ploshcha"
PROMPT = "Розкажи мені, що тут і до чого, поясни, як влаштований і як працює цей проект."
REPEATS = 3

FACTS = {
    "дизайн-док перед кодом": r"дизайн-док|ПЕРЕД кодом",
    "docs не в гіті": r"docs/.{0,40}(не в гіт|ігнор|gitignore)|gitignore.{0,40}docs",
    "не комітити самому": r"не коміт|лише за.{0,20}коміть|без явного",
    "uv sync і extra dev": r"uv sync|--extra dev",
    "без AI-атрибуції": r"атрибуц|co-authored|githooks",
    "замір перед фіксом": r"замір перед|перш ніж записати",
}


