import json
from pathlib import Path
from typing import Literal

from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

from src.datamodel import Document

ROOT_PATH = Path(__file__).parent.parent


def load_data(lang: Literal['eu', 'es']):
    with (ROOT_PATH / 'data' / lang / f'BASSE.{lang}.jsonl').open() as f:
        data = [Document.from_json(json.loads(line)) for line in f]
    return data


def load_rubric(criterion):
    with (ROOT_PATH / 'assets' / 'rubrics.json').open() as rf:
        rubrics = json.load(rf)
    rubric_data = rubrics[criterion]
    rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
    return rubric
