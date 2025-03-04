import pandas as pd
import argparse
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", default="eu", choices=["eu", "es"])
	args = parser.parse_args()


	data = []
	with open(os.path.join("..", args.lang, "summ.jsonl")) as infile:
		for line in infile:
			data.append(json.loads(line))

	model_data = []
	models = ['claude', 'commandr', 'gpt4o', 'reka', 'llama3']
	prompts = ["base", "cot", "5w1h", "tldr"]
	criteria = ["Coherence", "Consistency", "Fluency", "Relevance", "5W1H"]

	# Check human eval
	for human in ["human-ann1", "human-ann2", "human-ann3"]:
		example = {"model": human, "prompt": "None"}
		total = 0
		for criterion in criteria:
			anns = [np.average(e["model_summaries"][human]["anns"][criterion]) for e in data[:15]]
			avg = np.array(anns).mean()
			example[criterion] = avg
			total += avg
		example["avg."] = total / 5
		model_data.append(example)

	# Check baseline
	example = {"model": "subhead", "prompt": "None"}
	total = 0
	for criterion in criteria:
		anns = [np.average(e["model_summaries"]["subhead"]["anns"][criterion]) for e in data]
		avg = np.array(anns).mean()
		example[criterion] = avg
		total += avg
	example["avg."] = total / 5
	model_data.append(example)

	# Check model eval
	for model in models:
		for prompt in prompts:
			name = model + "-" + prompt
			example = {"model": model, "prompt": prompt}
			total = 0
			for criterion in criteria:
				anns = [np.average(e["model_summaries"][name]["anns"][criterion]) for e in data]
				avg = np.array(anns).mean()
				example[criterion] = avg
				total += avg
			example["avg."] = total / 5
			model_data.append(example)

	df = pd.DataFrame(model_data)

	os.makedirs("correlations", exist_ok=True)

	with open("correlations/" + args.lang + "_model_eval.txt", "w") as outfile:
		outfile.write(df.to_latex(index=False, float_format="%.2f"))