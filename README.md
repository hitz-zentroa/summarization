# BASSE: BAsque and Spanish Summarization Evaluation

This Github repository hosts the data and evaluation results for [Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans?](https//:anonymous.link.com). It additionally hosts BasqueSumm, the first collection of news documents and their corresponding subheads (often used as a proxy for summaries) for Basque.


## Table of contents:

1. [BASSE](#basse)
2. [Codebase](#codebase)
3. [BasqueSumm](#basquesumm)
4. [Licensing](#licensing)
5. [Citation](#citation)

## BASSE

BASSE is a multilingual (Basque and Spanish) dataset designed primarily for the metaevaluation of automatic summarization metrics and LLM-as-a-Judge models. We generate automatic summaries for 90 news documents in these two languages (45 each) using Anthropic's Claude, OpenAI's GPT-4, Reka AI's Reka, Llama3.1 and CommandR+. For each of these models, we use four different prompts (base, core, 5W1H, tldr: [see paper for more details](link-to-paper)), with the goal of generating a diverse array of summaries, both regarding quality and style. We also include human-generated reference summaries for each news document.

After generating these summaries, we annotate them for Coherence, Consistency, Fluency, Relevance, and 5W1H on a 5-point Likert scale, largely following the annotation protocol from [SummEval](https://github.com/Yale-LILY/SummEval).


### How to use BASSE

The BASSE dataset is contained in two jsonl files: eu/BASSE.jsonl and es/BASSE.jsonl. To load the data, use the following code snippet:

```
import json

basse_data = [json.loads(line) for line in open("eu/BASSE.jsonl")]
```


### BASSE Format

Each line is a json object with the following keys, value pairs:

* `'idx'`: (String) A unique identifier defined by the url of the original publication.

* `'round'`: (Int) 1, 2, or 3 - Which annotation round this example comes from: rounds 1 and 2 have 3 annotations per criteria and 3 human reference summaries per document, while round 3 only has a single annotation and reference summary.

* `'original_document'`: (String) The original news document to be summarized.

* `'reference_summaries'`: (List) The human-generated referece summaries. 3 per document in rounds 1 and 2, and a single reference summary per document for round 3.

* `'model_summaries'`: (Dict) The generated summary and its human annotations on a 5-point Likert scale for the 5 criteria: Coherence, Consistency, Fluency, Relevance, and 5W1H. The model key is a combination of model and prompt, e.g "claude-base". We also include annotations for the subhead baseline ("subhead") and similarly include the evaluations of the human reference summaries are included in rounds 1 and 2 ("human-ann1", "human-ann2", "human-ann3"). Note that these only have 2 annotations per summary.

For example, the first instance in es/BASSE.jsonl would contain the following information (note that we cut off long summaries and only include two examples of model summaries, while in reality there are 24 models: 5 models x 4 prompts + 3 human refs + 1 subhead):

```
{
	'idx': 'http://elpais.com/deportes/2019/08/17/actualidad/1566005143_044557.html'
	'round': 1
	'original_document': 'El jet lag ante Argentina , que quedó maquillado por el arrebato febril de Ricky ( 15 puntos en los...'
	'reference_summaries': ['La selección española pierde 55-74 contra ...',
	                        'La selección española de baloncesto perdió ante Rusia (55.74) ...',
	                        'El equipo español de baloncesto perdió...']
	'model_summaries': {'human-ann1': {'summ': 'La selección española pierde 55-74...',
	                                   'anns': {'Coherence': [5.0, 5.0],
	                                            'Consistency': [5.0, 5.0]},
	                                            'Fluency': [5.0, 5.0],
	                                            'Relevance': [5.0, 5.0],
	                                            '5W1H': [3.0, 4.0]}
	                                   },
	                    ...
	                    'claude-base': {'summ': 'El texto describe un partido de preparación de la selección española de baloncesto...',
	                                    'anns': {'Coherence': [2.0, 3.0, 3.0],
	                                             'Consistency': [4.0, 4.0, 5.0],
	                                             'Fluency': [5.0, 5.0, 5.0],
	                                             'Relevance': [4.0, 4.0, 3.0],
	                                             '5W1H': [5.0, 4.0, 5.0]}
	                                   }
	                    ...
	                    }
}
```


## Codebase

We base our experimental setup on the codebase from [SummEval](https://github.com/Yale-LILY/SummEval). To use our evaluation code, first clone our gitub repo and then install the SummEval subrepo in /experiments.

```
git clone 
```

To reproduce the correlations between the metrics and human annotations, use the following script:

```
cd experiments
python metrics_exp.py
```

To get the model evaluations, use this one instead:
```
cd experiments
python model_eval.py
```


## BasqueSumm

This dataset was automatically compiled from www.berria.eus using trafilatura to extract the texts.

Each line is a json object with the following keys, value pairs:

* `'date'`: (String) 'yyyy-mm-dd' - When the article was published.

* `'url'`: (String) The url of the original publication.

* `'category'`: (String) the articles topic, e.g., economy, society.

* `title'`: (String) The title of the article.

* `subtitle'`: (String) The subtitle of the article

* `summary'`: (String) The combined title + subtitle which acts as a proxy for a reference summary

* `'text'`: (String) The news article.




## Licensing

We release BASSE and BasqueSumm under a [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## Citation

If you use the baselines or data from this shared task, please cite the following paper, as well as the papers for the specific datasets that you use (see the bib files that follow afterwards) :

```
@misc{barnes-etal-2025-basse,
    title = "Summarization Metrics for Spanish and Basque: {D}o Automatic Scores and LLM-Judges Correlate with Humans?",
    author = "Barnes, Jeremy and
              Perez, Naiara and
              Bonet-Jover, Alba and
              Altuna, Begoña",
    year={2025},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={}
}
```