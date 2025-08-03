# BASSE: BAsque and Spanish Summarization Evaluation

This GitHub repository hosts the data, evaluation results, and codebase for 
[Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans?](https://arxiv.org/abs/2503.17039). 
It additionally hosts BasqueSumm, the first collection of news documents and their 
corresponding subheads (often used as a proxy for summaries) for Basque.

## Table of contents

1. [BASSE](#basse)
   * [How to use BASSE](#how-to-use-basse)
   * [BASSE format](#basse-format)
2. [Codebase](#codebase)
   * [Setup](#setup)
   * [0. Analysis of the BASSE corpus](#0-analysis-of-the-basse-corpus)
   * [1. Score summaries with automatic metrics](#1-score-summaries-with-automatic-metrics)
   * [2. Score summaries with judge LLMs](#2-score-summaries-with-judge-llms)
   * [3. Compute correlations](#3-compute-correlations-between-metricjudge-scores-and-human-ratings)
3. [BasqueSumm](#basquesumm)
4. [Licensing](#licensing)
5. [Citation](#citation)

## BASSE

BASSE is a multilingual (Basque and Spanish) dataset designed primarily for the 
**meta-evaluation of automatic summarization metrics and LLM-as-a-Judge models**. 
We generated automatic summaries for 90 news documents in these two languages (45 each) 
using Anthropic's **Claude**, OpenAI's **GPT-4o**, Reka AI's **Reka**, Meta's 
**Llama 3.1 Instruct** and Cohere's **Command R+**. For each of these models, we use four 
different prompts (**base**, **core**, **5W1H**, **tldr**; 
[see paper for more details](https://arxiv.org/abs/2503.17039)), with the goal of 
generating a diverse array of summaries, both regarding quality and style. We also include 
human-generated reference summaries for each news document.

After generating these summaries, we annotated them for **Coherence**, **Consistency**, 
**Fluency**, **Relevance**, and **5W1H** on a 5-point Likert scale, largely following the 
annotation protocol from [SummEval](https://github.com/Yale-LILY/SummEval).

### How to use BASSE

The BASSE dataset is contained in two JSONLines files: 
* [data/eu/BASSE.eu.jsonl](data/eu/BASSE.eu.jsonl)
* [data/es/BASSE.es.jsonl](data/es/BASSE.es.jsonl)

To load the data, use the following code snippet:
```python
import json

eu_data = [json.loads(line) for line in open("data/eu/BASSE.eu.jsonl")]
```

### BASSE format

Each line is a json object with the following keys, value pairs:

* `"idx"` (str): A unique identifier defined by the url of the original publication.
* `"round"` (int): `0`, `1`, `2`, or `3` - Which annotation round this example comes from: rounds `0`, `1` and `2` have three annotations per criteria and three human reference summaries per document, while round `3` only has a single annotation and reference summary. See section [Inter-annotator agreement](#inter-annotator-agreement).
* `"original_document"` (str): The original news document to be summarized.
* `"reference_summaries"` (list): The human-generated reference summaries. 3 per document in rounds 1 and 2, and a single reference summary per document for round 3.
* `"model_summaries"` (dict): The generated summary and its human annotations on a 5-point Likert scale for the 5 criteria: Coherence, Consistency, Fluency, Relevance, and 5W1H. The model key is a combination of model and prompt, e.g `"claude-base"`. We also include annotations for the subhead baseline (`"subhead"`) and similarly include the evaluations of the human reference summaries are included in rounds 1 and 2 (`"human-ann1"`, `"human-ann2"`, `"human-ann3"`). Note that these only have 2 annotations per summary.

For example, the first instance in [data/es/BASSE.es.jsonl](data/es/BASSE.es.jsonl) would contain the following information (note that we cut off long summaries and only include two examples of model summaries, while in reality there are 24 models: 5 models x 4 prompts + 3 human refs + 1 subhead):

```json
{
  "idx": "http://elpais.com/deportes/2019/08/17/actualidad/1566005143_044557.html",
  "round": 1,
  "original_document": "El jet lag ante Argentina , que quedó maquillado por el...",
  "reference_summaries": [
    "La selección española pierde 55-74 contra ...",
    "La selección española de baloncesto perdió ante Rusia (55.74) ...",
    "El equipo español de baloncesto perdió..."
  ],
  "model_summaries": {
    "human-ann1": {
      "summ": "La selección española pierde 55-74...",
      "anns": {
        "Coherence": [5.0, 5.0],
        "Consistency": [5.0, 5.0],
        "Fluency": [5.0, 5.0], 
        "Relevance": [5.0, 5.0], 
        "5W1H": [3.0, 4.0]
      }
    },
    ...
    "claude-base": {
      "summ": "El texto describe un partido de preparación...", 
      "anns": {
        "Coherence": [2.0, 3.0, 3.0],
        "Consistency": [4.0, 4.0, 5.0],
        "Fluency": [5.0, 5.0, 5.0],
        "Relevance": [4.0, 4.0, 3.0],
        "5W1H": [5.0, 4.0, 5.0]
      }
    },
    ...
  }
}
```

## Codebase

### Setup

We base our experimental setup on the codebases of
[SummEval](https://github.com/Yale-LILY/SummEval) for automatic metrics and
[Prometheus-Eval](https://github.com/prometheus-eval/prometheus-eval) for LLM-as-a-Judge 
evaluation.

To use this repo, first clone it and install the necessary dependencies:
```shell
git clone --recursive https://github.com/hitz-zentroa/summarization.git
cd summarization
pip install -r requirements.txt
```
The `--recursive` flag ensures that SummEval is cloned into the `lib` directory. If you 
prefer, you can instead attempt to install it directly via pip:

```shell
pip install summ-eval
```

#### Environment variables

If you're planning to run automatic metrics, you need to set the following environment 
variable (required by ROUGE):
```shell
export ROUGE_HOME=$PWD/lib/SummEval/evaluation/summ_eval/ROUGE-1.5.5
```

#### Required assets

You will also need to download a couple of assets. 

NLTK's `punkt_tab` resources are necessary to preprocess BASSE for automatic metric
scoring. Download them by doing:
```python
import nltk
nltk.download('punkt_tab')
```

FastText embeddings are used to compute ROUGE scores. Download them into [assets](assets):
```shell
wget -P assets https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.vec.gz
wget -P assets https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz
```

### 0. Analysis of the BASSE corpus

#### Inter-annotator agreement

We measured agreement between human annotators for each evaluation criterion in terms of
**Cohen's quadratic kappa**, **κ**, **Krippendorff's ordinal alpha**, **ɑ**, and 
**agreement percentage**.

Use the notebook [agreement.ipynb](src/notebooks/agreement.ipynb) to reproduce the 
tables and plots included in our article, which are located under 
[results/agreement](results/agreement).

Note that agreement was computed over three annotation rounds: rounds `0` and `1` involve 
annotations on the same set of summaries, before and after agreement discussions between 
annotators, respectively. Round `0` is provided separately from the BASSE corpus for 
reproducibility ([BASSE.es.round_0.jsonl](data/es/BASSE.es.round_0.jsonl) and 
[BASSE.eu.round_0.jsonl](data/eu/BASSE.eu.round_0.jsonl)). Round `2` covers a new set of 
summaries. In the paper, we refer to rounds `0`, `1`, and `2` as `1`, `2`, and 
`3`, respectively, as a matter of style.

#### Quantitative and qualitative analysis

The notebook [corpus.ipynb](src/notebooks/corpus.ipynb) contains code for both 
quantitative and qualitative analysis of the BASSE corpus.

The **quantitative analysis** includes computing the number of documents, tokens, and 
sentences per document, along with vocabulary size and the number of summaries generated 
in the target language. These metrics are aggregated by LLM, prompt, and language. 
Additionally, we use SummEval’s DataStats metric to assess summary compression and novel 
bigram usage.

The **qualitative analysis** reports human ratings on the criteria **Coherence**, 
**Consistency**, **Fluency**, **Relevance**, and **5W1H** broken down by LLM and prompt
pair.

All plots and tables generated by the notebook, as included in our paper, are available 
under [results/corpus](results/corpus).

### 1. Score summaries with automatic metrics

Following the [SummEval](https://github.com/Yale-LILY/SummEval) framework, we apply a 
range of standard automatic metrics (some with minor adaptations or fixes provided in 
[src/metrics](src/metrics)), to later meta-evaluate their correlation with human judgments 
in Spanish and Basque.

To reproduce the metric-based evaluations, run the script:
```shell
python -m src.01_apply_automatic_metrics --language LANGUAGE --metrics METRIC [METRIC ...]
```
where `LANGUAGE` is one of `{eu,es}` and `METRIC` is one of 
`{rouge,m_rouge_we,m_bert_score,bleu,chrf,m_meteor,cider,stats}` or `all` to compute all 
implemented metrics.

The output of this script is located under [pred/metrics](pred/metrics). It consists of 
one CSV file per language and metric, with the following columns:
* `model` (str): the model whose summaries have been evaluated. As with the BASSE corpus, it is a combination of the actual model name and prompt, e.g `"claude-base"`.
* `metric` (str): the name of the specific metric; `m_bert_score`, for instance, produces BertScore precision, recall and F1-score.
* `score` (float): the score obtained by the model's summaries (usually the mean of the individual summaries' scores, but the aggregation depends on the metric).

Each CSV contains one row per model name and prompt pair.

### 2. Score summaries with judge LLMs

Using the [Prometheus-Eval](https://github.com/prometheus-eval/prometheus-eval) codebase, 
we run a range of open and proprietary LLMs to evaluate summaries across five criteria, to 
later meta-evaluate their correlation with human judgments in Spanish and Basque.

To reproduce the judge-based evaluations, run the script:
```shell
python -m src.02_apply_llm_as_a_judge --model MODEL --language LANGUAGE --criterion CRITERION [--num_gpus NUM_GPUS] [--use_async_lite]
```
where `LANGUAGE` is one of `{eu,es}` and `CRITERION` is one of 
`{Coherence,Consistency,Fluency,Relevance,5W1H}`. `NUM_GPUS` controls the number of GPUs 
used by vLLM to load the judge locally. Use the flag `--use_async_lite` instead when 
using a commercial LLM API for judging (see supported providers here: 
https://docs.litellm.ai/docs/providers).

The output of this script is located under [pred/judges](pred/judges). It consists of one
CSV file per language, judge, and criterion, with the following columns:
* `model` (str): the model whose summaries have been evaluated. As with the BASSE corpus, it is a combination of the actual model name and prompt, e.g `"claude-base"`.
* `feedback` (str): the feedback given by the model to explain, support or justify the produced score.
* `score` (float): the score obtained by the summary.

Each CSV contains one row per summary judged.

### 3. Compute correlations between metric/judge scores and human ratings

To reproduce the correlations between the automatic and human evaluations, run the script:
```shell
python -m src.03_compute_correlations --language LANGUAGE [--metrics METRIC [METRIC ...]] [--judges JUDGE [JUDGE ...]]
```
where `LANGUAGE` is one of `{eu,es}`,  `METRIC` is the name of a metric in 
[pred/metrics](pred/metrics) or `all`, and `JUDGE` is the name of a judge in 
[pred/judges](pred/judges) or `all`.

The output of this script is located under [results/correlation](results/correlation). It
consists of one LaTeX table per language, evaluator type (metric or judge LLM), and 
correlation statistic (**Spearman's rank correlation coefficient**, **ρ**, and **Kendall's 
rank correlation coefficient**, **τ**).

Use the notebook [correlations.ipynb](src/notebooks/correlations.ipynb) to reproduce the 
correlations plot included in our article.

## BasqueSumm

[BasqueSumm](data/eu/berria_summ.jsonl) was automatically compiled from www.berria.eus 
using [trafilatura](https://trafilatura.readthedocs.io) to extract the texts.

Each line is a JSON object with the following keys, value pairs:

* `"date"` (str): When the article was published, formatted as `"yyyy-mm-dd"`.
* `"url"` (str): The URL of the original publication.
* `"category"` (str): the articles topic, e.g., economy, society.
* `"title"` (str): The title of the article.
* `"subtitle"` (str): The subtitle of the article.
* `"summary"` (str): The combined title + subtitle, which acts as a proxy for a reference summary.
* `"text"` (str): The news article.

## Licensing

We release BASSE and BasqueSumm under a [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Citation

Please cite the following paper if you use the BASSE corpus or its associated codebase:
```
@misc{barnes2025summarizationmetricsspanishbasque,
      title={Summarization Metrics for {S}panish and {B}asque: Do Automatic Scores and {LLM}-Judges Correlate with Humans?}, 
      author={Jeremy Barnes and Naiara Perez and Alba Bonet-Jover and Begoña Altuna},
      year={2025},
      eprint={2503.17039},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.17039}, 
}
```