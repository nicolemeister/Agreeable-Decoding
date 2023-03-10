import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import json
import pandas as pd
from os import path
from collections import defaultdict
from thresholding import optimal_threshold

parser = argparse.ArgumentParser()

parser.add_argument('--prompt_strength', default='strong', type=str) # normal or strong 
parser.add_argument('--model_num', default='2', type=int) # text-davinici 002 or 003
parser.add_argument('--temp', default='0.5', type=float)
parser.add_argument('--N', default=3, type=int) # text-davinici 002 or 003
parser.add_argument('--test_thresholds', default=False, type=bool) # text-davinici 002 or 003
parser.add_argument('--mini', default=False, type=bool) # text-davinici 002 or 003
parser.add_argument('--enforce_certainty', default=False, type=bool) # e.g. summarization, decoding


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
def eval(q_to_a, eval_results, data, incorrect_results, threshold, agreementscores, enforce_certainty):
    questions = data.keys()
    sum_correct_noprompt_incorrect,n_all_sum_correct_noprompt_incorrect = 0,0

    for question in questions:
        
        if enforce_certainty: only_question = question[5:-57]
        else: only_question = question[5:-8]
        outputs = data[question] # keys are question + 'include your best guess'
        ans = q_to_a[only_question]
        #try: ans = df.loc[[df.index[df['Question'] == only_question][0]]]['Answer'][0]
        #except: ans = df.loc[[df.index[df['Question'] == only_question][0]]]['Answer']
        agreementscore = agreementscores[question]['off_diagonal_gap']

        for original_prompt_type in outputs.keys():
            if '_logprob' not in original_prompt_type: 
                # do thresholding if not no prompt
                if original_prompt_type == 'noprompt': thresholded_prompt_type='noprompt'
                else: 
                    if agreementscore<threshold: thresholded_prompt_type='noprompt' # if below threshold (ie not a polarizing question)
                    else: thresholded_prompt_type=original_prompt_type # if above threshold (ie a polarizing question), keep persona 

                # go through all the outputs to identify  1. at least one ans is correct  2. all ans are correct 3. total correct   
                # 4. at least one ans is DK  # 5. all ans are DK # 6. total DK 
                correct_bools = []
                DK_bools = []
                for output in outputs[thresholded_prompt_type]:
                    if original_prompt_type=='SUM': 
                        all_sum_correct_noprompt_incorrect = []
                        for output_noprompt in outputs['noprompt']:
                            if match(ans, output) and not match(ans, output_noprompt):
                                # print("\nQuestion: {} \n Ans: {} \n SUM_output: {} \n Noprompt_Output: {}".format(question, ans, output, output_noprompt))
                                sum_correct_noprompt_incorrect+=1
                                all_sum_correct_noprompt_incorrect.append(True)
                            else: all_sum_correct_noprompt_incorrect.append(False)
                        if (sum(all_sum_correct_noprompt_incorrect)==3): 
                            print("\nQuestion: {} \n Ans: {} \n SUM_output: {} \n Noprompt_Output1: {} \nNoprompt_Output2: {} \nNoprompt_Output3: {} \n".format(question, ans, output, outputs['noprompt'][0],outputs['noprompt'][1],outputs['noprompt'][2]))
                            n_all_sum_correct_noprompt_incorrect+=1

                    eval_results[original_prompt_type]['total_questions']+=1
                    
                    if match(ans, output):
                        eval_results[original_prompt_type]['total_correct']+=1
                        correct_bools.append(True)
                    else: 
                        incorrect_results[original_prompt_type].append(question+str(ans))
                        incorrect_results[original_prompt_type+"_questions"].append(question)
                        correct_bools.append(False)

                    if match("don't know", output) or match("do not know", output):
                        eval_results[original_prompt_type]['total_DK']+=1
                        DK_bools.append(True)
                    else: DK_bools.append(False)
                    # only do this once (if you are thresholding based on summarize)
                    if 'SUM' in original_prompt_type:
                        break

                if any(correct_bools): eval_results[original_prompt_type]['atleastone_correct']+=1
                if any(DK_bools): eval_results[original_prompt_type]['atleastone_DK']+=1
                if all(correct_bools): eval_results[original_prompt_type]['all_correct']+=1
                if all(DK_bools): eval_results[original_prompt_type]['all_DK']+=1
                if (any(correct_bools) and any(DK_bools)): eval_results[original_prompt_type]['correct_and_dk']+=1


    print('sum_correct_noprompt_incorrect: ', sum_correct_noprompt_incorrect)
    print('n_correct_noprompt_incorrect: ', n_all_sum_correct_noprompt_incorrect)
    return eval_results, incorrect_results

# read in trivia QA: create dictionary mapping questions to answers
df = pd.read_excel('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/triviaQA_subset.xlsx')
q_to_a = dict(zip(df['Question'], df['Answer']))

if args.enforce_certainty: prompt_stength_addon='_enforce_certainty'
else: prompt_stength_addon=''

# read LM outputs from prompts (merge the two files)
if not args.mini: 
    path_triviaQA = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}{}/triviaQA/temp_{}/'.format(args.model_num, args.prompt_strength, prompt_stength_addon, args.temp)
path_triviaQAmini = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}{}/triviaQAmini/temp_{}/'.format(args.model_num, args.prompt_strength, prompt_stength_addon, args.temp)
path_polarizingqs = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}/polarizingqs/temp_{}/'.format(args.model_num, args.prompt_strength, args.temp)
path_polarizingqs_large = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}/polarizingqs_large/temp_{}/'.format(args.model_num, args.prompt_strength, args.temp)


# need to account for case where one of the dem gets it but not all 

eval_results_data = {'total_questions': 0, 'correct_and_dk': 0, 
                    'atleastone_correct': 0, 'all_correct': 0,  'total_correct': 0, 
                    'atleastone_DK': 0, 'all_DK': 0,  'total_DK': 0}

triviaQAmini_scores_path=path_triviaQAmini+"score_N_{}.json".format(args.N)
if not args.mini: triviaQA_scores_path=path_triviaQA+"score_N_{}.json".format(args.N)
polarizingqs_scores_path=path_polarizingqs+"score_N_{}.json".format(args.N)
polarizingqs_large_scores_path=path_polarizingqs_large+"score_N_{}.json".format(args.N)


if not args.mini: 
    with open(path_triviaQA+"N_{}.json".format(args.N), 'r') as f: triviaQA_results = json.load(f)
with open(path_triviaQAmini+"N_{}.json".format(args.N), 'r') as f: triviaQAmini_results = json.load(f)

if not args.mini: 
    with open(triviaQA_scores_path, 'r') as f: triviaQA_agreementscore = json.load(f)
with open(triviaQAmini_scores_path, 'r') as f: triviaQAmini_agreementscore = json.load(f)

if not args.mini: optimal_t, trivia_offdiaggap_polarizing, _ = optimal_threshold(polarizingqs_scores_path, polarizingqs_large_scores_path, triviaQAmini_scores_path, triviaQA_scores_path)
else: optimal_t, trivia_offdiaggap_polarizing, _ = optimal_threshold(polarizingqs_scores_path, polarizingqs_large_scores_path, triviaQAmini_scores_path)
print("optimal_t: ", optimal_t)

if args.test_thresholds:  
    thresholds = [-0.05]
    for i in np.linspace(0, 75, 76):
        thresholds.append(np.percentile(trivia_offdiaggap_polarizing, i))
    # thresholds.append(optimal_t)
else: thresholds = [optimal_t]

combined_results = {} # map threshold to total_correct_answers/total_questions

for threshold in thresholds:
    incorrect_results = defaultdict(list) # store question and answers that are incorrect 
    combined_results[threshold]= defaultdict(dict)
    print(threshold)
    eval_results = {'dem': eval_results_data.copy(),
                'repub': eval_results_data.copy(),
                'noprompt': eval_results_data.copy(),
                'SUM': eval_results_data.copy(),
                'SUM_noquestion': eval_results_data.copy(),
                'SUM_tatsu': eval_results_data.copy()
                }
    eval_results, incorrect_results = eval(q_to_a, eval_results=eval_results, data=triviaQAmini_results, incorrect_results=incorrect_results, threshold=threshold, agreementscores=triviaQAmini_agreementscore, enforce_certainty=args.enforce_certainty)
    if not args.mini: eval_results, incorrect_results = eval(q_to_a, eval_results=eval_results, data=triviaQA_results, incorrect_results=incorrect_results, threshold=threshold, agreementscores=triviaQA_agreementscore, enforce_certainty=args.enforce_certainty)

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

        combined_results[threshold][prompt_type]['total_correct_%'] = eval_results[prompt_type]['total_correct_%']
        combined_results[threshold][prompt_type]['atleastone_correct_%'] = eval_results[prompt_type]['atleastone_correct_%']

    if threshold==optimal_t: is_optimal_t = True
    else: is_optimal_t=False
    evalresults_path = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}{}/triviaQA_eval_t_{:.04f}_optimalt_{}.json'.format(args.model_num, args.prompt_strength, prompt_stength_addon, threshold, is_optimal_t)

    # Eval: check for substring match, check standard eval for trivia qa for free text
    out_file = open(evalresults_path, "w")
    json.dump(eval_results, out_file, indent = 6)

    print("Saving results in {}".format(evalresults_path))
    incorrect_results['dem_incorrect_noprompt_correct'] = list(set(incorrect_results['dem_questions']).difference(set(incorrect_results[''])))
    incorrect_results['repub_incorrect_noprompt_correct'] = list(set(incorrect_results['repub_questions']).difference(set(incorrect_results['noprompt_questions'])))

    incorrect_results['dem_correct_noprompt_incorrect'] = list(set(incorrect_results['noprompt_questions']).difference(set(incorrect_results['dem_questions'])))
    incorrect_results['repub_correct_noprompt_incorrect'] = list(set(incorrect_results['noprompt_questions']).difference(set(incorrect_results['repub_questions'])))

    incorrect_results_path = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}{}/incorrect.json'.format(args.model_num, args.prompt_strength, prompt_stength_addon)
    out_file = open(incorrect_results_path, "w")
    json.dump(incorrect_results, out_file, indent = 6)

combinedresults_path = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}{}/triviaQA_eval_combined.json'.format(args.model_num, args.prompt_strength, prompt_stength_addon)
print("Saving results in {}".format(combinedresults_path))
out_file = open(combinedresults_path, "w")
json.dump(combined_results, out_file, indent = 6)