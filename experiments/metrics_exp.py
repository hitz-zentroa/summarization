import pandas as pd
from summ_eval.rouge_metric import RougeMetric
from summ_eval.bleu_metric import BleuMetric
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.data_stats_metric import DataStatsMetric

import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

rouge = RougeMetric()
bertscore = BertScoreMetric()
#moverscore = MoverScoreMetric()
bleu = BleuMetric()
chrf = ChrfppMetric()
meteor = MeteorMetric()
cider = CiderMetric()
datastats = DataStatsMetric()



models = ['🤖 Claude Sonnet 3.5', '🤖 Command R+', '🤖 GPT 4o', '🤖 Reka Core', '🤖 Llama-3.1-70b-instruct']
prompts = ["Base", "CoT", "5W1H", "tldr"]
criteria = ["Coherence", "Consistency", "Fluency", "Relevance", "5W1H"]
rouge_metrics = ["rouge_1_f_score", "rouge_2_f_score", "rouge_3_f_score", 
				 "rouge_4_f_score", "rouge_l_f_score", "rouge_su*_f_score"]
bertscore_metrics = ['bert_score_precision', 'bert_score_recall', 'bert_score_f1']
datastats_metrics = ["summary_length", "percentage_novel_1-gram", 
                     "percentage_novel_2-gram", "percentage_novel_3-gram",
                     "percentage_repeated_1-gram_in_summ",
                     "percentage_repeated_2-gram_in_summ",
                     "percentage_repeated_3-gram_in_summ",
                     "coverage", "compression", "density"]

metric_name_map = {"rouge_1_f_score": "ROUGE-1",
                    "rouge_2_f_score": "ROUGE-2",
                    "rouge_3_f_score": "ROUGE-3",
                    "rouge_4_f_score": "ROUGE-4",
                    "rouge_l_f_score": "ROUGE-L",
                    "rouge_su*_f_score": "ROUGE-su*",
				    'bert_score_precision': "BertScore-p",
				    'bert_score_recall': "BertScore-r",
				    'bert_score_f1': "BertScore-f",
				    "summary_length": "Length",
				    "percentage_novel_1-gram": "Novel unigram", 
                    "percentage_novel_2-gram": "Novel bi-gram",
                    "percentage_novel_3-gram": "Novel tri-gram",
                    "percentage_repeated_1-gram_in_summ": "Repeated unigram",
                    "percentage_repeated_2-gram_in_summ": "Repeated bi-gram",
                    "percentage_repeated_3-gram_in_summ": "Repeated tri-gram",
                    "coverage": "Stats-coverage",
                    "compression": "Stats-compression",
                    "density": "Stats-density",
                    "bleu": "BLEU",
                    "chrf": "CHRF",
                    "meteor": "METEOR",
                    "cider": "CIDEr"
				 }


files = ["../eu/round1/summarisation-poc-EU-jeremy.xlsx", "../eu/round1/summarisation-poc-EU-begoña.xlsx"]


human_x = {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}
model_x = {}

human_raw = {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}
model_raw = {}

# Open single annotation document
xls = pd.ExcelFile(files[0], engine="openpyxl")

# Get my summaries and original documents
df1= pd.read_excel(xls, '🖐🏾 Human')
refs = list(df1[df1["Author"] == "Heading"]["Summary"])
original_docs = list(df1["Text"].dropna())

xls = pd.ExcelFile(files[0], engine="openpyxl")

for model in models:
	for prompt in prompts:
		# Get a single models summaries
		df2= pd.read_excel(xls, model)
		summs = list(df2[df2["Prompt"] == prompt]["Summary"])

		# Evaluate rouge metrics
		d = rouge.evaluate_batch(summs, refs)
		for metric in rouge_metrics:
			if metric not in model_x:
				model_x[metric] = [d["rouge"][metric]]
			else:
				model_x[metric].append(d["rouge"][metric])

		# Rouge WE 
		# need to get dependency embeddings for Spanish/Basque and add a language option to original code??


		# S3
		# Requires lots of language specific resources and training

		# BERTscore, prec, rec, f1
		# works fine with multilingual bert, but is quite slow
		
		d = bertscore.evaluate_batch(summs, refs)
		for metric in bertscore_metrics:
			if metric not in model_x:
				model_x[metric] = [d[metric]]
			else:
				model_x[metric].append(d[metric])


		# MoverScore - uses distilbert, need to update for Spanish/Basque
		"""
		d = moverscore.evaluate_batch(summs, refs)
		for metric in bertscore_metrics:
			if metric not in model_x:
				model_x[metric] = [d[metric]]
			else:
				model_x[metric].append(d[metric])
		"""

		# SMS


		# BLEU
		d = bleu.evaluate_batch(summs, refs)
		for metric in d.keys():
			if metric not in model_x:
				model_x[metric] = [d[metric]]
			else:
				model_x[metric].append(d[metric])


		# CHRF
		d = chrf.evaluate_batch(summs, refs)
		for metric in d.keys():
			if metric not in model_x:
				model_x[metric] = [d[metric]]
			else:
				model_x[metric].append(d[metric])

		# Meteor
		d = meteor.evaluate_batch(summs, refs)
		for metric in d.keys():
			if metric not in model_x:
				model_x[metric] = [d[metric]]
			else:
				model_x[metric].append(d[metric])

		# Cider
		d = cider.evaluate_batch(summs, refs)
		for metric in d.keys():
			if metric not in model_x:
				model_x[metric] = [d[metric]]
			else:
				model_x[metric].append(d[metric])

		# Datastats
		d = datastats.evaluate_batch(summs, original_docs)
		for metric in datastats_metrics:
			if metric not in model_x:
				model_x[metric] = [d[metric]]
			else:
				model_x[metric].append(d[metric])


		# Get the average human annotation for one criterion over model+prompt
		# We follow SummEval and Louis and Nenkova (2013) and report Kendall's tau correlations
		# between automatic metrics and human judgements calculated at system level
		# For each system, we average the human judgements over all summaries for each criterion
		for criterion in criteria:
			human = np.average(df2[df2["Prompt"] == prompt][criterion])
			human_x[criterion].append(human)

# now gather kendall tau for all combinations of criteria and metric
tau_data = np.zeros((len(model_x),len(criteria)))

for i, metric in enumerate(model_x.keys()):
	model = model_x[metric]
	for j, criterion in enumerate(criteria):
		hum = human_x[criterion]
		tau, p_value = stats.kendalltau(hum, model)
		tau_data[i,j] = tau
		s, p_value = stats.spearmanr(hum, model)
		spear_data[i,j] = s

# print results
print("Tau correlations")
df = pd.DataFrame(tau_data, columns=criteria).round(3)
metric_names = [metric_name_map[m] for m in model_x.keys()]
df.insert(0, "Metric", metric_names)
print(df.to_string(index=False))