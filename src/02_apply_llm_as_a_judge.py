import argparse
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from prometheus_eval import PrometheusEval
from prometheus_eval.litellm import AsyncLiteLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT
from prometheus_eval.vllm import VLLM

from src.constants import Criterion, SummaryConfig
from src.datamodel import Document
from src.util import load_data, load_rubric, ROOT_PATH

logging.getLogger().setLevel(logging.INFO)


def prepare_data(data: list[Document]):
    # transform data into Prometheus-input-like
    summary_configs, instructions, responses, references = [], [], [], []
    for summary_config in list(SummaryConfig):
        for doc in data:
            instructions.append(f'Summarize the following text:\n\n{doc.original_document}')
            responses.append(doc.model_summaries[summary_config].summ)
            references.append(doc.reference_summaries)
            summary_configs.append(summary_config)
    return instructions, responses, references, summary_configs


def main(model: str, lang: Literal['eu', 'es'], criterion: Criterion, num_gpus: int = 2, use_async_lite: bool = False):

    if use_async_lite:
        judge_llm = AsyncLiteLLM(model, requests_per_minute=10)
    else:
        judge_llm = VLLM(model=model, tensor_parallel_size=num_gpus)
    judge = PrometheusEval(model=judge_llm, absolute_grade_template=ABSOLUTE_PROMPT)

    instructions, responses, reference_answers, summary_configs = prepare_data(load_data(lang))
    logging.info(f'Imported {len(responses)} documents for evaluation')
    rubric = load_rubric(criterion)

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=rubric,
        reference_answers=reference_answers  # TODO: have a look at this; it expects only be one ref per summary?
    )

    for summary_config, feedback, score in zip(summary_configs, feedbacks, scores):
        logging.debug('Feedback:', feedback)
        logging.debug('Score:', score)

    df = pd.DataFrame(list(zip(summary_configs, feedbacks, scores)), columns=['model', 'feedback', 'score'])
    pred_path = ROOT_PATH / 'pred' / 'judges' / lang / Path(model).name
    pred_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_path / f'{criterion}.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--language', choices=['eu', 'es'], required=True)
    parser.add_argument('--criterion', choices=list(Criterion), required=True)
    parser.add_argument('--num_gpus', default=2)
    parser.add_argument('--use_async_lite', action='store_true')
    args = parser.parse_args()

    logging.info(args)

    main(args.model, args.language, args.criterion, args.num_gpus, args.use_async_lite)
