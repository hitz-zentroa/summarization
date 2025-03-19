from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from prometheus_eval.vllm import VLLM
import pandas as pd
import json
import argparse
import os
import torch

def import_data(criteria, language="eu"):
    models = ['claude-base', 'claude-core', 'claude-5w1h', 'claude-tldr', 'commandr-base', 'commandr-core', 'commandr-5w1h', 'commandr-tldr', 'gpt4o-base', 'gpt4o-core', 'gpt4o-5w1h', 'gpt4o-tldr', 'reka-base', 'reka-core', 'reka-5w1h', 'reka-tldr', 'llama3-base', 'llama3-core', 'llama3-5w1h', 'llama3-tldr']

    summ_data = [json.loads(line) for line in open("../../" + language + "/BASSE.jsonl")]

    # keep track of models for each one
    model = []    

    # instructions
    instructions = []

    # responses
    responses = []

    # reference answers
    references = []


    for m in models:
        instructions.extend(["Summarize the following text:\n\n {}".format(doc["original_document"]) for doc in summ_data])
        responses.extend([doc["model_summaries"][m]["summ"] for doc in summ_data])
        references.extend([ref["reference_summaries"] for ref in summ_data])
        for ref in summ_data:
            model.append(m)

    # rubric
    with open("summ_data/rubrics.json") as o:
        rubrics = json.load(o)
    rubric_data = rubrics[criteria]
    rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
    return instructions, responses, references, rubric, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="eu")
    parser.add_argument("--crit", default="Coherence")
    parser.add_argument("--model", default="prometheus-eval/prometheus-7b-v2.0")
    parser.add_argument("--num_gpus", default=2)
    args = parser.parse_args()


    model = VLLM(model=args.model, tensor_parallel_size=args.num_gpus)
    judge = PrometheusEval(
        model=model,
        absolute_grade_template=ABSOLUTE_PROMPT,
    )


    instructions, responses, reference_answers, rubric, models = import_data(args.crit, args.lang)
    print("imported {} documents for evaluation".format(len(responses)))

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=rubric,
        reference_answers=reference_answers
    )

    for model, feedback, score in zip(models, feedbacks, scores):
        print("Feedback:", feedback)
        print("Score:", score)

    df = pd.DataFrame(list(zip(models, feedbacks, scores)), columns=["model", "feedback", "score"])


    model_name = os.path.dirname(args.model)
    path = os.path.join("results", args.lang, model_name)
    os.makedirs(path, exist_ok=True)
    outfile = os.path.join(path, args.crit + ".csv")
    df.to_csv(outfile, index=False)