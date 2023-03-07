import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import json
import pandas as pd
from os import path
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--prompt_strength', default='strong', type=str) # normal or strong 
parser.add_argument('--model_num', default='2', type=int) # text-davinici 002 or 003
parser.add_argument('--temp', default='0.5', type=float)
parser.add_argument('--N', default=3, type=int) # text-davinici 002 or 003

args = parser.parse_args()

# check for substring match
def match(ans, output):
    ans = str(ans)
    output = str(output)
    return (ans.lower() in output.lower())

# Input:
# - the data of the ouptuts with (dem, rep, no prompt, summarization outputs) : 
#   - /Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/002/strong/triviaQAmini/temp_0.5/N_3.json
#   - /Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/002/strong/triviaQA/temp_0.5/N_3.json
# - trivia qa answers : /Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/triviaQA_subset.xlsx

# input dictionary of eval_results (for results to be filled in) and data (with the model's outputs)
# return eval_results with 
def eval(q_to_a, eval_results, data, incorrect_results):
    questions = data.keys()
    for question in questions:
        
        only_question = question[5:-8]
        outputs = data[question]
        ans = q_to_a[only_question]
        #try: ans = df.loc[[df.index[df['Question'] == only_question][0]]]['Answer'][0]
        #except: ans = df.loc[[df.index[df['Question'] == only_question][0]]]['Answer']

        for prompt_type in outputs.keys():
            if '_logprob' not in prompt_type: 
                # go through all the outputs to identify  1. at least one ans is correct  2. all ans are correct 3. total correct   
                # 4. at least one ans is DK  # 5. all ans are DK # 6. total DK 
                correct_bools = []
                DK_bools = []
                for output in outputs[prompt_type]:
                    eval_results[prompt_type]['total_questions']+=1
                    
                    if match(ans, output):
                        eval_results[prompt_type]['total_correct']+=1
                        correct_bools.append(True)
                    else: 
                        incorrect_results[prompt_type].append(question+str(ans))
                        incorrect_results[prompt_type+"_questions"].append(question)
                        correct_bools.append(False)

                    if match("don't know", output):
                        eval_results[prompt_type]['total_DK']+=1
                        DK_bools.append(True)
                    else: DK_bools.append(False)

                if any(correct_bools): eval_results[prompt_type]['atleastone_correct']+=1
                if any(DK_bools): eval_results[prompt_type]['atleastone_DK']+=1
                if all(correct_bools): eval_results[prompt_type]['all_correct']+=1
                if all(DK_bools): eval_results[prompt_type]['all_DK']+=1
                if (any(correct_bools) and any(DK_bools)): eval_results[prompt_type]['correct_and_dk']+=1
    return eval_results, incorrect_results

# read in trivia QA: create dictionary mapping questions to answers
df = pd.read_excel('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/triviaQA_subset.xlsx')
q_to_a = dict(zip(df['Question'], df['Answer']))

# read LM outputs from prompts (merge the two files)
path_triviaQAmini = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}/triviaQAmini/temp_{}/N_{}.json'.format(args.model_num, args.prompt_strength, args.temp, args.N)
path_triviaQA = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}/triviaQA/temp_{}/N_{}.json'.format(args.model_num, args.prompt_strength, args.temp, args.N)

# need to account for case where one of the dem gets it but not all 

eval_results_data = {'total_questions': 0, 'correct_and_dk': 0, 
                    'atleastone_correct': 0, 'all_correct': 0,  'total_correct': 0, 
                    'atleastone_DK': 0, 'all_DK': 0,  'total_DK': 0}

eval_results = {'dem': eval_results_data.copy(),
                'repub': eval_results_data.copy(),
                'noprompt': eval_results_data.copy(),
                'SUM': eval_results_data.copy(),
                'SUM_noquestion': eval_results_data.copy(),
                'SUM_tatsu': eval_results_data.copy()
                }

with open(path_triviaQA, 'r') as f: triviaQA_results = json.load(f)
with open(path_triviaQAmini, 'r') as f: triviaQAmini_results = json.load(f)

incorrect_results = defaultdict(list) # store question and answers that are incorrect 

eval_results, incorrect_results = eval(q_to_a, eval_results=eval_results, data=triviaQAmini_results, incorrect_results=incorrect_results)
print('here1')
eval_results, incorrect_results = eval(q_to_a, eval_results=eval_results, data=triviaQA_results, incorrect_results=incorrect_results)
print('here2')
print(eval_results.keys())

for prompt_type in eval_results.keys():
    if 'SUM' not in prompt_type: N=3
    else: N=1
    eval_results[prompt_type]['atleastone_correct_%']=(eval_results[prompt_type]['atleastone_correct']/eval_results[prompt_type]['total_questions'])*N
    eval_results[prompt_type]['atleastone_DK_%']=(eval_results[prompt_type]['atleastone_DK']/eval_results[prompt_type]['total_questions'])*N
    eval_results[prompt_type]['all_correct_%']=(eval_results[prompt_type]['all_correct']/eval_results[prompt_type]['total_questions'])*N
    eval_results[prompt_type]['all_DK_%']=(eval_results[prompt_type]['all_DK']/eval_results[prompt_type]['total_questions'])*N
    eval_results[prompt_type]['correct_and_dk_%']=(eval_results[prompt_type]['correct_and_dk']/eval_results[prompt_type]['total_questions'])*N
    eval_results[prompt_type]['total_correct_%']=eval_results[prompt_type]['total_correct']/eval_results[prompt_type]['total_questions']
    eval_results[prompt_type]['total_DK_%']=eval_results[prompt_type]['total_DK']/eval_results[prompt_type]['total_questions']


evalresults_path = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}/triviaQA_eval.json'.format(args.model_num, args.prompt_strength)
# Eval: check for substring match, check standard eval for trivia qa for free text
out_file = open(evalresults_path, "w")
json.dump(eval_results, out_file, indent = 6)

incorrect_results['dem_incorrect_noprompt_correct'] = list(set(incorrect_results['dem_questions']).difference(set(incorrect_results[''])))
incorrect_results['repub_incorrect_noprompt_correct'] = list(set(incorrect_results['repub_questions']).difference(set(incorrect_results['noprompt_questions'])))

incorrect_results['dem_correct_noprompt_incorrect'] = list(set(incorrect_results['noprompt_questions']).difference(set(incorrect_results['dem_questions'])))
incorrect_results['repub_correct_noprompt_incorrect'] = list(set(incorrect_results['noprompt_questions']).difference(set(incorrect_results['repub_questions'])))

incorrect_results_path = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}/incorrect.json'.format(args.model_num, args.prompt_strength)
out_file = open(incorrect_results_path, "w")
json.dump(incorrect_results, out_file, indent = 6)