import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import pickle
import json
import sys
import random
import os
import os.path
from os import path
from collections import defaultdict
import pandas as pd
from decoding_summarization import get_text_logprob

sys.path.append('helm')
sys.path.append('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/helm/src')
from src.helm.common.authentication import Authentication
from src.helm.proxy.services.remote_service import RemoteService, RemoteServiceError
from src.helm.common.request import Request

parser = argparse.ArgumentParser(description='LLM Plausibility Set')

parser.add_argument('--model_output_dir', default='/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results')
parser.add_argument('--num_workers', default=15, type=int)
parser.add_argument('--chunk_size', default=1, type=int)
parser.add_argument('--question_type', default='polarizingqs', type=str) # polarizingqs or neutralqs
parser.add_argument('--prompt_strength', default='strong', type=str) # normal or strong 
parser.add_argument('--model_num', default='2', type=int) # text-davinici 002 or 003
parser.add_argument('--temp', default='0.5', type=float)
parser.add_argument('--N', default=3, type=int) # text-davinici 002 or 003
parser.add_argument('--input_data', default='summarization', type=str) # e.g. summarization, decoding


args = parser.parse_args()

def add_ex_to_prompt(prompt, questions, answers):
    for idx, i in enumerate(range(7, 7+len(questions))):
        prompt +=  "## Example {}: ".format(str(i)) + '\n' + '\n'+\
                   "### Question {}: ".format(str(i)) + '\n' +questions[idx] + '\n' + '\n' + \
                   "### Answer {}: ".format(str(i)) + '\n' +answers[idx] + '\n' + '\n'

    prompt+="## Correct answers for examples 7-16:".format(str(7+len(questions)-1))+ '\n' + '\n' + "### Likert score for example 7:"
    return prompt

# given an output e.g., (\n3. Acceptable answer with minor imperfections.\n\n### Likert score for example 8:\n3. Acceptable answer with minor imperfections.\n\n### Likert score for example 9:\n3. Acceptable answer with minor imperfections.\n\n### Likert score for example 10:\n4. Informative and satisfying answer.\n\n### Likert score for example 11:\n4. Informative and satisfying answ)
# return a list of answers (likert score ratings)
def parse(output):
    output=output.split('\n')
    # deal with first output that doesn't contain instruction before e.g., '### Likert score for example 8"
    answers = [output[1][0]]
    for i, value in enumerate(output):
        if '### Likert score for example' in value:
            answers.append(output[i+1][0])
    return answers

# Prompt 003 for subjectivity and informativity rating of model outputs

# for each model there are X types of outputs
# unprompted
# democrat persona
# republican persona
# with summarization trick (if 002)

# read in the informative instruction prompt 
with open('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/helm/prompt_informative.txt') as f:
    lines = f.readlines()
informative_prompt_instructions = ''.join(lines)

# read in the subjective instruction prompt 
with open('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/helm/prompt_subjective.txt') as f:
    lines = f.readlines()
subjective_prompt_instructions = ''.join(lines)

# extract questions and their answers 
path = "../crfm_results/summarization/00{}/{}/{}/temp_{}/".format(args.model_num, args.prompt_strength, args.question_type, args.temp)
# read in the data polarizingqs_N_3.json
with open("{}/N_{}.json".format(path, args.N), 'r') as f:
    result_gen = json.load(f)

questions = list(result_gen.keys())

labels_gen = ['dem', 'repub', 'noprompt', 'SUM', 'SUM_noquestion', 'SUM_tatsu']
label_to_N = {'dem': 3, 'repub': 3, 'noprompt': 3, 'SUM': 1, 'SUM_noquestion': 1, 'SUM_tatsu': 1}

prev_arg_model_num = args.model_num

results = defaultdict(dict)
for question in questions:
    answers, answer_order = [], []
    for label in labels_gen:
        # [gen] for each persona (dem, repub, SUM), grab the output S from P_Persona
        N=label_to_N[label]
        for i in range(N):
            ans = result_gen[question][label][i]
            answers.append(ans)
            answer_order.append(label)
    
    results[question]['answer_order'] = answer_order
    args.model_num = '3'

    # INFORMATIVE
    final_informative_prompt = add_ex_to_prompt(prompt=informative_prompt_instructions, questions=[question for i in range(len(answers))], answers=answers)
    informative_output, _ = get_text_logprob(args, final_informative_prompt)
    informative_output = parse(informative_output) # parse output
    results[question]['answers_informative'] = informative_output
    print(informative_output)

    # SUBJECTIVE
    final_subjective_prompt = add_ex_to_prompt(prompt=subjective_prompt_instructions, questions=[question for i in range(len(answers))], answers=answers)
    subjective_output, _ = get_text_logprob(args, final_subjective_prompt)
    subjective_output = parse(subjective_output)
    results[question]['answers_subjective'] = subjective_output
    print(subjective_output)


args.model_num = prev_arg_model_num
# save the data 
path = "../crfm_results/summarization/00{}/{}/{}/temp_{}/".format(args.model_num, args.prompt_strength, args.question_type, args.temp)
if not os.path.exists(path): os.makedirs(path)
out_file = open("{}/subjective_informative.json".format(path, args.N), "w")
json.dump(results, out_file, indent = 6)
