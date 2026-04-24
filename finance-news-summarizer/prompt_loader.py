from pathlib import Path
from string import Template


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt_text(filename):
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def render_prompt(filename, **kwargs):
    return Template(load_prompt_text(filename)).substitute(**kwargs)
