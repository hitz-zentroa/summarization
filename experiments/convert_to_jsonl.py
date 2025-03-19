import json
import pandas as pd
import argparse
import os

models = ['🤖 Claude Sonnet 3.5', '🤖 Command R+', '🤖 GPT 4o', '🤖 Reka Core', '🤖 Llama-3.1-70b-instruct']
round3_models = ['🤖 Claude Haiku 3.5', '🤖 Command R+', '🤖 GPT 4o', '🤖 Reka Flash', '🤖 Llama-3.1-70b-instruct']
prompts = {"Base": "base", "CoT": "core", "5W1H": "5w1h", "tldr": "tldr"}
criteria = ["Coherence", "Consistency", "Fluency", "Relevance", "5W1H"]

name_map = {'🤖 Claude Sonnet 3.5': "claude",
		 	'🤖 Claude Haiku 3.5': "claude",
            '🤖 Command R+': "commandr",
            '🤖 GPT 4o': "gpt4o",
            '🤖 Reka Core': "reka",
            '🤖 Reka Flash': "reka",
            '🤖 Llama-3.1-70b-instruct':"llama3"}

annotator_mapping = {"eu": {"ann1": "Naiara", "ann2": "Begoña", "ann3": "Jeremy"},
                     "es": {"ann1": "Alba", "ann2": "Begoña", "ann3": "Jeremy"}}


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", default="eu", choices=["eu", "es"])
	args = parser.parse_args()

	if args.lang == "eu":
		ann_files = {"ann1": [pd.ExcelFile("../eu/round1-post-discussion/summarisation-poc-EU-naiara.xlsx", engine="openpyxl"),
						  pd.ExcelFile("../eu/round2/summarisation-poc-EU-naiara.xlsx", engine="openpyxl")],
	             "ann2": [pd.ExcelFile("../eu/round1-post-discussion/summarisation-poc-EU-begoña.xlsx", engine="openpyxl"),
						  pd.ExcelFile("../eu/round2/summarisation-poc-EU-begoña.xlsx", engine="openpyxl")],
	             "ann3": [pd.ExcelFile("../eu/round1-post-discussion/summarisation-poc-EU-jeremy.xlsx", engine="openpyxl"),
						  pd.ExcelFile("../eu/round2/summarisation-poc-EU-jeremy.xlsx", engine="openpyxl")]
				}
		round3 = {"ann1": pd.ExcelFile("../eu/round3/summarisation-poc-naiara.xlsx", engine="openpyxl"),
	              "ann2": pd.ExcelFile("../eu/round3/summarisation-poc-begoña.xlsx", engine="openpyxl"),
	              "ann3": pd.ExcelFile("../eu/round3/summarisation-poc-jeremy.xlsx", engine="openpyxl")}
	elif args.lang == "es":
		ann_files = {"ann1": [pd.ExcelFile("../es/round1-post-discussion/summarisation-poc-ES-alba.xlsx", engine="openpyxl"),
						  pd.ExcelFile("../es/round2/summarisation-poc-ES-alba.xlsx", engine="openpyxl")],
	             "ann2": [pd.ExcelFile("../es/round1-post-discussion/summarisation-poc-ES-begoña.xlsx", engine="openpyxl"),
						  pd.ExcelFile("../es/round2/summarisation-poc-ES-begoña.xlsx", engine="openpyxl")],
	             "ann3": [pd.ExcelFile("../es/round1-post-discussion/summarisation-poc-ES-jeremy.xlsx", engine="openpyxl"),
						  pd.ExcelFile("../es/round2/summarisation-poc-ES-jeremy.xlsx", engine="openpyxl")]
				}
		round3 = {"ann1": pd.ExcelFile("../es/round3/summarisation-poc-ES-alba.xlsx", engine="openpyxl"),
	              "ann2": pd.ExcelFile("../es/round3/summarisation-poc-ES-begoña.xlsx", engine="openpyxl"),
	              "ann3": pd.ExcelFile("../es/round3/summarisation-poc-ES-jeremy.xlsx", engine="openpyxl")}



	data = []

	# Extract human summaries
	dfs = [pd.read_excel(xls, '🖐🏾 Human')for xls in ann_files["ann3"]]
	df = pd.concat(dfs)
	oidxs = [idx.strip() for idx in df["ID"].dropna()]

	refs1 = list(df[df["Author"] == annotator_mapping[args.lang]["ann1"]]["Summary"])
	refs2 = list(df[df["Author"] == annotator_mapping[args.lang]["ann2"]]["Summary"])
	refs3 = list(df[df["Author"] == annotator_mapping[args.lang]["ann3"]]["Summary"])
	refs = [list(summaries) for summaries in zip(refs1, refs2, refs3)]
	original_docs = list(df["Text"].dropna())

	assert len(oidxs) == len(refs) == len(original_docs)

	for i, (idx, ref, orig) in enumerate(zip(oidxs, refs, original_docs)):
		if i < 10:
			r = 1
		else:
			r = 2
		data.append({"idx": idx, "round": r, "original_document": orig, "reference_summaries": ref, "model_summaries": {}})



	for ann, xls in round3.items():
		df = pd.read_excel(xls, '🖐🏾 Human')
		idxs = [idx.strip() for idx in df["ID"].dropna()]
		r3_refs = df[df["Author"] == annotator_mapping[args.lang][ann]]["Summary"]
		r3_refs = [[l] for l in r3_refs]
		refs = r3_refs
		original_docs = list(df["Text"].dropna())
		
		for idx, ref, orig in zip(idxs, refs, original_docs):
			data.append({"idx": idx, "round": 3, "original_document": orig, "reference_summaries": ref, "model_summaries": {}})

	# iterate over the human summaries
	for ref_ann, ref_name in annotator_mapping[args.lang].items():

		# get the annotations from the other annotators
		for ann in annotator_mapping[args.lang].keys():
			if ref_ann != ann:
				name = 'human-' + ref_ann
				docs = ann_files[ann]
				dfs = []
				for i, xls in enumerate(docs):
					dfs.append(pd.read_excel(xls, '🖐🏾 Human'))

				# get the dataframe from the specific annotator
				df = pd.concat(dfs)

				
				idxs = [idx.strip() for idx in df["ID"].dropna()]
				summs = list(df[df["Author"] == ref_name]["Summary"])
				
				for i ,(idx, summ) in enumerate(zip(idxs, summs)):
					
					assert idxs[i] == data[i]["idx"]

					info = {"summ": summ, "anns": {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}}
					if name not in data[i]["model_summaries"]:
						data[i]["model_summaries"][name] = info

				# collect the annotations for each criterion in criteria
				for criterion in criteria:
					
					annotations = list(df[df["Author"] == ref_name][criterion])
					assert len(idxs) == len(annotations)
					for i, annotation in enumerate(annotations):
						data[i]["model_summaries"][name]["anns"][criterion].append(annotation)



	# get model evals
	for model in models:
		for prompt, pname in prompts.items():
			name = name_map[model] + '-' + pname
			
			for ann, docs in ann_files.items():
				dfs = []
				for i, xls in enumerate(docs):
					try:
						dfs.append(pd.read_excel(xls, model))
					except ValueError:
						print("{0} is missing annotations for {1}-{2}  Round{3}".format(ann, model, prompt, i+1))
				# get the dataframe for combined annotations of each annotator
				df = pd.concat(dfs)

				# we only need to add the model summaries once
				if ann == "ann1":
					idxs = [idx.strip() for idx in df["ID"].dropna()]
					model_summs = list(df[df["Prompt"] == prompt]["Summary"])
					for i ,(idx, summ) in enumerate(zip(idxs, model_summs)):
						assert idxs[i] == data[i]["idx"]
						info = {"summ": summ, "anns": {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}}
						if name not in data[i]["model_summaries"]:
							data[i]["model_summaries"][name] = info

				# collect the annotations for each criterion in criteria
				
				for criterion in criteria:
					annotations = list(df[df["Prompt"] == prompt][criterion])
					assert len(idxs) == len(annotations)
					for i, annotation in enumerate(annotations):
						data[i]["model_summaries"][name]["anns"][criterion].append(annotation)
			
	# Round 3 is a bit different - 2 models changed and only one annotator per doc
	for model in round3_models:
		for prompt, pname in prompts.items():
			name = name_map[model] + '-' + pname

			# need to keep track of the annotators to deal with the idx
			for i, (ann, xls) in enumerate(round3.items()):
				try:
					df = pd.read_excel(xls, model)
				except ValueError:
					print("{0} is missing annotations for {1}-{2}  Round3".format(ann, model, prompt))


				idxs = [idx.strip() for idx in df["ID"].dropna()]
				model_summs = list(df[df["Prompt"] == prompt]["Summary"].dropna())
				for j ,(idx, summ) in enumerate(zip(idxs, model_summs)):
					offset = 15 + i*10 + j
					assert idxs[j] == data[offset]["idx"]
					info = {"summ": summ, "anns": {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}}
					if name not in data[offset]["model_summaries"]:
						data[offset]["model_summaries"][name] = info


				# collect the annotations for each criterion in criteria
				for criterion in criteria:
					annotations = list(df[df["Prompt"] == prompt][criterion].dropna())
					assert len(idxs) == len(annotations)

					for j, annotation in enumerate(annotations):
						offset = 15 + i*10 + j
						assert idxs[j] == data[offset]["idx"]
						data[offset]["model_summaries"][name]["anns"][criterion].append(annotation)

			

	# Get the baseline results
	for ann, docs in ann_files.items():
		dfs = []
		for i, xls in enumerate(docs):
			try:
				dfs.append(pd.read_excel(xls, '🖐🏾 Human'))
			except ValueError:
				print("{0} is missing annotations for {1}  Round{3}".format(ann, "subhead", i+1))
		# get the dataframe for combined annotations of each annotator
		df = pd.concat(dfs)

		# we only need to add the model summaries once
		if ann == "ann1":
			idxs = [idx.strip() for idx in df["ID"].dropna()]
			model_summs = list(df[df["Author"] == "Heading"]["Summary"])
			for i ,(idx, summ) in enumerate(zip(idxs, model_summs)):
				assert idxs[i] == data[i]["idx"]
				info = {"summ": summ, "anns": {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}}
				if "subhead" not in data[i]["model_summaries"]:
					data[i]["model_summaries"]["subhead"] = info

		# collect the annotations for each criterion in criteria
		
		for criterion in criteria:
			annotations = list(df[df["Author"] == "Heading"][criterion])
			assert len(idxs) == len(annotations)
			for i, annotation in enumerate(annotations):
				data[i]["model_summaries"]["subhead"]["anns"][criterion].append(annotation)
			
	# Round 3 is a bit different - 2 models changed and only one annotator per doc

	# need to keep track of the annotators to deal with the idx
	for i, (ann, xls) in enumerate(round3.items()):
		try:
			df = pd.read_excel(xls, '🖐🏾 Human')
		except ValueError:
			print("{0} is missing annotations for {1} Round3".format(ann, 'subhead'))


		idxs = [idx.strip() for idx in df["ID"].dropna()]
		model_summs = list(df[df["Author"] == "Heading"]["Summary"].dropna())
		for j ,(idx, summ) in enumerate(zip(idxs, model_summs)):
			offset = 15 + i*10 + j
			assert idxs[j] == data[offset]["idx"]
			info = {"summ": summ, "anns": {"Coherence": [], "Consistency": [], "Fluency": [], "Relevance": [], "5W1H": []}}
			if "subhead" not in data[offset]["model_summaries"]:
				data[offset]["model_summaries"]["subhead"] = info


		# collect the annotations for each criterion in criteria
		for criterion in criteria:
			annotations = list(df[df["Author"] == "Heading"][criterion].dropna())
			assert len(idxs) == len(annotations)

			for j, annotation in enumerate(annotations):
				offset = 15 + i*10 + j
				assert idxs[j] == data[offset]["idx"]
				data[offset]["model_summaries"]["subhead"]["anns"][criterion].append(annotation)


	# check that all the data is correct
	for i, example in enumerate(data):
		assert len(example["original_document"]) > 0
		if i < 15:
			assert len(example["reference_summaries"]) == 3
			assert len(example["model_summaries"]) == 24
		else:
			assert len(example["reference_summaries"]) == 1
			assert len(example["model_summaries"]) == 21


	# save as a jsonl file
	outname = os.path.join("..", args.lang, "BASSE.jsonl")
	with open(outname, "w") as outfile:
		for example in data:
			json.dump(example, outfile)
			outfile.write("\n")