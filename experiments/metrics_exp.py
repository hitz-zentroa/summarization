import pandas as pd
from summ_eval.rouge_metric import RougeMetric
from summ_eval.multilingual_rouge_we_metric import MultilingualRougeWeMetric
from summ_eval.bleu_metric import BleuMetric
from summ_eval.bert_score_metric import BertScoreMetric
#from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.multilingual_meteor_metric import MultilingualMeteorMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.data_stats_metric import DataStatsMetric

import snowballstemmer

from nltk import tokenize

import argparse
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", default="eu", choices=["eu", "es"])
	parser.add_argument("--metrics", nargs="+", default=["all"])
	parser.add_argument("--judges", nargs="+", default=["prometheus"])
	args = parser.parse_args()

	
	if args.metrics == ["all"]:
		metrics = metrics = ["rouge", "mrougewe", "mbertscore", "bleu", "chrf", "mmeteor", "cider", "stats"]
	else:
		metrics = args.metrics

	print("Metrics to calculate: {}".format(" ".join(metrics)))

	
	if args.lang == "eu":
		emb_path = "/home/jeremy/Exps/summarization/experiments/SummEval/evaluation/summ_eval/embeddings/cc.eu.300.vec.gz"
		stemmer = snowballstemmer.stemmer("basque")
	else:
		emb_path = "/home/jeremy/Exps/summarization/experiments/SummEval/evaluation/summ_eval/embeddings/cc.es.300.vec.gz"
		stemmer = snowballstemmer.stemmer("spanish")

	
	# Load all metrics except mrougewe

	# need to fix rouge to work without stemming or using already stemmed tokens
	#rouge = RougeMetric(rouge_args="-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a") # we use Rouge without the Porter stemmer, as it only works for English
	rouge = RougeMetric()
	bertscore = BertScoreMetric()
	bleu = BleuMetric()
	chrf = ChrfppMetric()
	meteor = MeteorMetric()
	mmeteor = MultilingualMeteorMetric()
	cider = CiderMetric()

	# we will use our own tokenized text instead of using spacy
	datastats = DataStatsMetric(tokenize=False)


	models = ['claude', 'commandr', 'gpt4o', 'reka', 'llama3']
	prompts = ["base", "cot", "5w1h", "tldr"]
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
	                    "multilingual_rouge_we_3_f": "mROUGE-we",
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
	                    "mmeteor": "MultilingualMETEOR",
	                    "cider": "CIDEr"
					 }




	human_x = {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}
	model_x = {}


	data = []
	for line in open(os.path.join("..", args.lang, "summ.jsonl")):
		data.append(json.loads(line))
	# Get summaries and original documents
	# will need to get summaries from all annotators at some point

	refs = [example["reference_summaries"] for example in data]
	original_docs = [example["original_document"] for example in data]

	# get vocab for multilingual rouge we
	vocab = set()
	for docs in refs:
		for d in docs:
			vocab.update(tokenize.word_tokenize(d.lower()))
	for model in models:
		for prompt in prompts:
			# Get a single models summaries
			name = model + '-' + prompt
			summs = [example["model_summaries"][name]["summ"] for example in data]
			for d in summs:
				vocab.update(tokenize.word_tokenize(d.lower()))

	

	# We instantiate the MultilingualRougeWeMetric at this point after getting the vocab
	# to reduce the number of word embeddings we import
	if "mrougewe" in metrics:
		mrwe = MultilingualRougeWeMetric(emb_path=emb_path, lang=args.lang, vocab=vocab)


	# tokenize and lowercase refs
	tokenized_refs = [[" ".join(tokenize.word_tokenize(summ.lower())) for summ in ref] for ref in refs]

	for model in models:
		for prompt in prompts:
			# Get a single models summaries
			name = model + '-' + prompt
			summs = [example["model_summaries"][name]["summ"] for example in data]

			# tokenize and lowercase candidate summaries
			tokenized_summs = [" ".join(tokenize.word_tokenize(summ.lower())) for summ in summs]


			# stem everything
			stemmed_summs = [" ".join(stemmer.stemWords(summ.split())) for summ in tokenized_summs]
			stemmed_refs = [[" ".join(stemmer.stemWords(summ.split())) for summ in ref] for ref in tokenized_refs]

			# Evaluate rouge metrics
			# Currently uses English stemmer -- need to fix
			# Need to tokenize and lowercase text
			if "rouge" in metrics:

				d = rouge.evaluate_batch(tokenized_summs, tokenized_refs)
				for metric in rouge_metrics:
					if metric not in model_x:
						model_x[metric] = [d["rouge"][metric]]
					else:
						model_x[metric].append(d["rouge"][metric])
	
			# Rouge WE
			if "mrougewe" in metrics: 
				
				d = mrwe.evaluate_batch(tokenized_summs, tokenized_refs)
				metric = "multilingual_rouge_we_3_f"
				if metric not in model_x:
					model_x[metric] = [d[metric]]
				else:
					model_x[metric].append(d[metric])
				

			# BERTscore, prec, rec, f1
			# works fine with multilingual bert base uncased, but is quite slow
			if "mbertscore" in metrics:
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
			# doesn't work right now because of reliance on glove embeddings taken from spacy or elmo


			# BLEU
			if "bleu" in metrics:
				d = bleu.evaluate_batch(tokenized_summs, tokenized_refs)
				for metric in d.keys():
					if metric not in model_x:
						model_x[metric] = [d[metric]]
					else:
						model_x[metric].append(d[metric])


			# CHRF
			if "chrf" in metrics:
				d = chrf.evaluate_batch(summs, refs)
				for metric in d.keys():
					if metric not in model_x:
						model_x[metric] = [d[metric]]
					else:
						model_x[metric].append(d[metric])

			# Multilingual Meteor
			if "mmeteor" in metrics:
				# the stems are likely not in the embedding space
				d = mmeteor.evaluate_batch(tokenized_summs, tokenized_refs)
				if "mmeteor" not in model_x:
					model_x["mmeteor"] = [d["meteor"]]
				else:
					model_x["mmeteor"].append(d["meteor"])

			# CIDEr
			if "cider" in metrics:
				d = cider.evaluate_batch(stemmed_summs, stemmed_refs)
				for metric in d.keys():
					if metric not in model_x:
						model_x[metric] = [d[metric]]
					else:
						model_x[metric].append(d[metric])

			# Dataset statistics
			if "stats" in metrics:
				# doesn't support multi-reference
				# the stats metric needs a list of lists = tokenized summaries
				stat = {'summary_length': 0,
				 'compression': 0,
				 'density': 0,
				 'percentage_novel_3-gram': 0,
				 'percentage_novel_2-gram': 0,
				 'coverage': 0,
				 'percentage_novel_1-gram': 0,
				 'percentage_repeated_1-gram_in_summ': 0,
				 'percentage_repeated_2-gram_in_summ': 0,
				 'percentage_repeated_3-gram_in_summ': 0}
				tok_summs = [l.split() for l in tokenized_summs]
				tok_refs = [[l.split() for l in ref] for ref in tokenized_refs]
				num_refs = len(tok_refs[0])
				# we perform this 3 times to take into account the 3 reference summaries in the first 15 documents
				for i in range(num_refs):
					indv_refs = [refs[i] for refs in tok_refs[:15]] + [refs[0] for refs in tok_refs[15:]]
					d = datastats.evaluate_batch(tok_summs, indv_refs)
					for metric in d:
						stat[metric] += d[metric]
				for metric in stat.keys():
					stat[metric] /= num_refs


				for metric in datastats_metrics:
					if metric not in model_x:
						model_x[metric] = [stat[metric]]
					else:
						model_x[metric].append(stat[metric])

	 
			# Get the average human annotation for one criterion over model+prompt
			# We follow SummEval and Louis and Nenkova (2013) and report Kendall's tau correlations
			# between automatic metrics and human judgements calculated at system level
			# For each system, we average the human judgements over all summaries for each criterion
			for criterion in criteria:
				annotation = []
				for example in data:
					# first we average all annotators over each example
					annotation.append(np.average(example["model_summaries"][name]["anns"][criterion]))
				# second we average all annotations over each model
				human_x[criterion].append(np.average(annotation))

	# now gather kendall tau for all combinations of criteria and metric
	tau_data = np.zeros((len(model_x), len(criteria)))
	spear_data = np.zeros((len(model_x), len(criteria)))

	for i, metric in enumerate(model_x.keys()):
		model = model_x[metric]
		for j, criterion in enumerate(criteria):
			hum = human_x[criterion]
			tau, p_value = stats.kendalltau(hum, model)
			tau_data[i,j] = tau
			s, p_value = stats.spearmanr(hum, model)
			spear_data[i,j] = s

	models = ['claude-base', 'claude-cot', 'claude-5w1h', 'claude-tldr', 'commandr-base', 'commandr-cot', 'commandr-5w1h', 'commandr-tldr', 'gpt4o-base', 'gpt4o-cot', 'gpt4o-5w1h', 'gpt4o-tldr', 'reka-base', 'reka-cot', 'reka-5w1h', 'reka-tldr', 'llama3-base', 'llama3-cot', 'llama3-5w1h', 'llama3-tldr']

	# print results for Kendall's Tau
	print("Tau correlations")
	df = pd.DataFrame(tau_data, columns=criteria).round(3)
	metric_names = [metric_name_map[m] for m in model_x.keys()]
	#metric_names.append("Prometheus")
	df.insert(0, "Metric", metric_names)
	print(df.to_string(index=False))

	with open("correlations/" + args.lang + "_tau_corr.txt", "w") as outfile:
		outfile.write(df.to_latex(index=False, float_format="%.3f"))


	# print results for Spearman correlation
	print("Spearman correlations")
	df = pd.DataFrame(spear_data, columns=criteria).round(3)
	metric_names = [metric_name_map[m] for m in model_x.keys()]
	#metric_names.append("Prometheus")
	df.insert(0, "Metric", metric_names)
	print(df.to_string(index=False))

	with open("correlations/" + args.lang + "_spear_corr.txt", "w") as outfile:
		outfile.write(df.to_latex(index=False, float_format="%.3f"))



	# now gather kendall tau / spearman for all combinations of criteria and judge
	tau_data = np.zeros((len(args.judges), len(criteria)))
	spear_data = np.zeros((len(args.judges), len(criteria)))
		
	for i, judge in enumerate(args.judges):
		for j, criterion in enumerate(criteria):
			judge_x = []
			hum = human_x[criterion]
			for model in models:	
				df = pd.read_csv("judge_results/eu/" + judge + "/" + criterion + ".csv")
				ave = df.loc[df['model'] == model]["score"].mean()
				judge_x.append(ave)
			tau, p_value = stats.kendalltau(hum, judge_x)
			tau_data[i,j] = tau
			s, p_value = stats.spearmanr(hum, judge_x)
			spear_data[i,j] = s
			#print(tau)


	# print results for Kendall's Tau
	print("Tau correlations")
	df = pd.DataFrame(tau_data, columns=criteria).round(3)
	df.insert(0, "Judge", args.judges)
	print(df.to_string(index=False))

	with open("correlations/" + args.lang + "_judge_tau_corr.txt", "w") as outfile:
		outfile.write(df.to_latex(index=False, float_format="%.3f"))
	

	print("Judge Spearman correlations")
	df = pd.DataFrame(spear_data, columns=criteria).round(3)
	df.insert(0, "Judge", args.judges)
	print(df.to_string(index=False))

	with open("correlations/" + args.lang + "_judge_spear_corr.txt", "w") as outfile:
		outfile.write(df.to_latex(index=False, float_format="%.3f"))