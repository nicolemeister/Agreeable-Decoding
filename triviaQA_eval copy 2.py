import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import json
import pandas as pd
from os import path
from collections import defaultdict
from thresholding import optimal_threshold
from collections import Counter
import re
import string
import matplotlib.pyplot as plt
from thresholding import roc_curve


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(ground_truth, prediction):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match(prediction, ground_truth):
    return prediction == ground_truth

# check for substring match
def substring_match(ans, output):
    ans = normalize_answer(str(ans))
    output = normalize_answer(str(output))
    return (ans in output)

# Input:
# - the data of the ouptuts with (dem, rep, no prompt, summarization outputs) : 
#   - /Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/002/strong/triviaQAmini/temp_0.5/N_3.json
#   - /Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/002/strong/triviaQA/temp_0.5/N_3.json
# - trivia qa answers : /Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/triviaQA_subset.xlsx

# input dictionary of eval_results (for results to be filled in) and data (with the model's outputs)
# return eval_results with 
# eval type is substring_match, f1, exact match
def eval(q_to_a, eval_results, data, incorrect_results, threshold, agreementscores, enforce_certainty, eval_type='substring_match'):
    questions = data.keys()
    sum_correct_noprompt_incorrect,n_all_sum_correct_noprompt_incorrect = 0,0

    for question in questions:
        
        if enforce_certainty: only_question = question[5:-57]
        else: only_question = question[5:-8]
        outputs = data[question] # keys are question + 'include your best guess'
        ans = normalize_answer(str(q_to_a[only_question]))
        #try: ans = df.loc[[df.index[df['Question'] == only_question][0]]]['Answer'][0]
        #except: ans = df.loc[[df.index[df['Question'] == only_question][0]]]['Answer']
        agreementscore = agreementscores[question]['off_diagonal_gap']
        
        for original_prompt_type in eval_results.keys():
            if '_logprob' not in original_prompt_type: 
                # do thresholding if not no prompt
                if original_prompt_type == 'noprompt': thresholded_prompt_type='noprompt'
                else: 
                    if agreementscore<threshold: thresholded_prompt_type='noprompt' # if below threshold (ie not a polarizing question)
                    else: thresholded_prompt_type=original_prompt_type # if above threshold (ie a polarizing question), keep persona 

                # go through all the outputs to identify  1. at least one ans is correct  2. all ans are correct 3. total correct   
                # 4. at least one ans is DK  # 5. all ans are DK # 6. total DK 
                substring_match_correct_bools, exact_match_correct_bools = [], []
                substring_match_DK_bools = []
                for output in outputs[thresholded_prompt_type]:
                    output = normalize_answer(str(output))
                    if original_prompt_type=='SUM': 
                        all_sum_correct_noprompt_incorrect = []
                        for output_noprompt in outputs['noprompt']:
                            output_noprompt = normalize_answer(str(output_noprompt))
                            if substring_match(ans, output) and not substring_match(ans, output_noprompt):
                                # print("\nQuestion: {} \n Ans: {} \n SUM_output: {} \n Noprompt_Output: {}".format(question, ans, output, output_noprompt))
                                sum_correct_noprompt_incorrect+=1
                                all_sum_correct_noprompt_incorrect.append(True)
                            else: all_sum_correct_noprompt_incorrect.append(False)
                        if (sum(all_sum_correct_noprompt_incorrect)==3): 
                            # print("\nQuestion: {} \n Ans: {} \n SUM_output: {} \n Noprompt_Output1: {} \nNoprompt_Output2: {} \nNoprompt_Output3: {} \n".format(question, ans, output, outputs['noprompt'][0],outputs['noprompt'][1],outputs['noprompt'][2]))
                            n_all_sum_correct_noprompt_incorrect+=1

                    eval_results[original_prompt_type]['total_questions']+=1
                    
                    # SUBSTRING MATCH
                    if substring_match(ans, output):
                        eval_results[original_prompt_type]['total_correct']+=1
                        substring_match_correct_bools.append(True)
                    else: 
                        incorrect_results[original_prompt_type].append(question+str(ans))
                        incorrect_results[original_prompt_type+"_questions"].append(question)
                        substring_match_correct_bools.append(False)

                    if substring_match("don't know", output) or substring_match("do not know", output):
                        eval_results[original_prompt_type]['total_DK']+=1
                        substring_match_DK_bools.append(True)
                    else: substring_match_DK_bools.append(False)
                    # only do this once (if you are thresholding based on summarize)

                    if substring_match("don't know", output) or substring_match("do not know", output):
                        eval_results[original_prompt_type]['total_DK']+=1
                        substring_match_DK_bools.append(True)
                    else: substring_match_DK_bools.append(False)
                    # only do this once (if you are thresholding based on summarize)

                    # EXACT MATCH
                    if exact_match(ans, output):
                        eval_results[original_prompt_type]['exact_match_total_correct']+=1
                        exact_match_correct_bools.append(True)
                    else:
                        exact_match_correct_bools.append(False)
                    # F1 SCORE 
                    eval_results[original_prompt_type]['f1']+=f1_score(ans, output)

                    if 'SUM' in original_prompt_type:
                        break
                
                # substring match
                if any(substring_match_correct_bools): eval_results[original_prompt_type]['atleastone_correct']+=1
                if any(substring_match_DK_bools): eval_results[original_prompt_type]['atleastone_DK']+=1
                if all(substring_match_correct_bools): eval_results[original_prompt_type]['all_correct']+=1
                if all(substring_match_DK_bools): eval_results[original_prompt_type]['all_DK']+=1
                if (any(substring_match_correct_bools) and any(substring_match_DK_bools)): eval_results[original_prompt_type]['correct_and_dk']+=1

                # exact match
                if any(exact_match_correct_bools): eval_results[original_prompt_type]['exact_match_atleastone_correct']+=1
                if all(exact_match_correct_bools): eval_results[original_prompt_type]['exact_match_all_correct']+=1

                
    #print('sum_correct_noprompt_incorrect: ', sum_correct_noprompt_incorrect)
    #print('n_correct_noprompt_incorrect: ', n_all_sum_correct_noprompt_incorrect)
    return eval_results, incorrect_results


def triviaQAeval(args, prompts, path):

    # read in trivia QA: create dictionary mapping questions to answers
    df = pd.read_excel('{}/../triviaQA_subset.xlsx'.format(args.model_output_dir))
    q_to_a = dict(zip(df['Question'], df['Answer']))

    if args.enforce_certainty: prompt_stength_addon='_enforce_certainty'
    else: prompt_stength_addon=''

    # read LM outputs from prompts (merge the two files)
    if not args.mini: 
        path_triviaQA = '{}/summarization/002/{}{}/triviaQA/temp_{}/'.format(args.model_output_dir, args.prompt_strength, prompt_stength_addon, args.temp)
    path_triviaQAmini = '{}/summarization/002/{}{}/triviaQAmini/temp_{}/'.format(args.model_output_dir, args.prompt_strength, prompt_stength_addon, args.temp)
    # path_polarizingqs = '{}/summarization/00{}/{}/{}/temp_{}/'.format(args.model_output_dir, args.model_num, args.prompt_strength, args.temp)
    # path_polarizingqs_large = '{}/summarization/00{}/{}/polarizingqs_large/temp_{}/'.format(args.model_output_dir, args.model_num, args.prompt_strength, args.temp)

    path_polarizingqs = path
    # need to account for case where one of the dem gets it but not all 

    eval_results_data = {'total_questions': 0, 'f1': 0, 'correct_and_dk': 0, 
                        'atleastone_correct': 0, 'all_correct': 0,  'total_correct': 0, 
                        'exact_match_atleastone_correct': 0, 'exact_match_all_correct': 0,  'exact_match_total_correct': 0, 
                        'atleastone_DK': 0, 'all_DK': 0,  'total_DK': 0}

    triviaQAmini_scores_path=path_triviaQAmini+"/score_N_{}.json".format(args.N)
    if not args.mini: triviaQA_scores_path=path_triviaQA+"score_N_{}.json".format(args.N)
    polarizingqs_scores_path=path_polarizingqs+"/score_N_{}.json".format(args.N)
    # polarizingqs_large_scores_path=path_polarizingqs_large+"score_N_{}.json".format(args.N)


    if not args.mini: 
        with open(path_triviaQA+"N_{}.json".format(args.N), 'r') as f: triviaQA_results = json.load(f)
    with open(path_triviaQAmini+"N_{}.json".format(args.N), 'r') as f: triviaQAmini_results = json.load(f)

    if not args.mini: 
        with open(triviaQA_scores_path, 'r') as f: triviaQA_agreementscore = json.load(f)
    with open(triviaQAmini_scores_path, 'r') as f: triviaQAmini_agreementscore = json.load(f)

    if not args.mini: optimal_t, trivia_offdiaggap, polarizing_offdiaggap, polarizing_offdiaggap_dict = optimal_threshold(polarizingqs_scores_path, triviaQAmini_scores_path, triviaQA_scores_path)
    else: optimal_t, trivia_offdiaggap, polarizing_offdiaggap, polarizing_offdiaggap_dict = optimal_threshold(polarizingqs_scores_path, triviaQAmini_scores_path)
    print("optimal_t: ", optimal_t)

    if args.test_thresholds:  
        thresholds = [-0.05]
        for i in np.linspace(0, 74, 25):
            thresholds.append(np.percentile(polarizing_offdiaggap, i))
        # thresholds.append(optimal_t)
    else: thresholds = [optimal_t]

    # PLOT ROC curve
    # y_true is a 0 or 1 label where 1 is polarizing question and 0 is nonpolarizing question (triviaQA) 
    y_true = np.concatenate((np.ones(len(polarizing_offdiaggap)), np.zeros(len(trivia_offdiaggap))), axis=0)
    # y_prob is the agreement scores, append these together, first with polarizing agreement scores then nonpolarizing agreement scores
    y_prob = np.concatenate((polarizing_offdiaggap,trivia_offdiaggap), axis=0)
    [fpr, tpr] = roc_curve(y_true, y_prob, thresholds)
    fpr, tpr = np.array(fpr), np.array(tpr)
    
    fig1 = plt.figure("Figure 1")
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("{}/ROC.jpg".format(path), dpi=1200)
    

    fig2 = plt.figure("Figure 2")
    # PLOT Agreement Metric
    plt.scatter(np.arange(len(trivia_offdiaggap)), trivia_offdiaggap, label='TriviaQA',s=3)
    plt.scatter(np.arange(len(polarizing_offdiaggap)), polarizing_offdiaggap, label='polarizing qs',s=3)

    plt.axhline(y = optimal_t, color = 'r', linestyle = '--', label='optimal threshold {:.04f}'.format(optimal_t))
    plt.legend(loc='upper right')
    plt.ylabel('Agreement metric')
    plt.savefig("{}/agreement_metric.jpg".format(path), dpi=1200)

    if list(prompts.keys())[0] in ['dem', 'repub']:
        combined_results = {} # map threshold to total_correct_answers/total_questions

        for threshold in thresholds:
            incorrect_results = defaultdict(list) # store question and answers that are incorrect 
            combined_results[threshold]= defaultdict(dict)
            print(threshold)
            # eval_results = {'noprompt': eval_results_data.copy(),
            #             'SUM': eval_results_data.copy(),
            #             'SUM_noquestion': eval_results_data.copy(),
            #             'SUM_tatsu': eval_results_data.copy()}
            eval_results = {'noprompt': eval_results_data.copy(),
                        'SUM': eval_results_data.copy()}

            for prompt in prompts:
                eval_results[prompt] = eval_results_data.copy()

            # SUBSTRING match
            eval_results, incorrect_results = eval(q_to_a, eval_results=eval_results, data=triviaQAmini_results, incorrect_results=incorrect_results, threshold=threshold, agreementscores=triviaQAmini_agreementscore, enforce_certainty=args.enforce_certainty)
            if not args.mini: eval_results, incorrect_results = eval(q_to_a, eval_results=eval_results, data=triviaQA_results, incorrect_results=incorrect_results, threshold=threshold, agreementscores=triviaQA_agreementscore, enforce_certainty=args.enforce_certainty)


            for prompt_type in eval_results.keys():
                if 'SUM' not in prompt_type: N=3
                else: N=1
                # SUBSTRING match
                eval_results[prompt_type]['atleastone_correct_%']=(eval_results[prompt_type]['atleastone_correct']/eval_results[prompt_type]['total_questions'])*N
                eval_results[prompt_type]['atleastone_DK_%']=(eval_results[prompt_type]['atleastone_DK']/eval_results[prompt_type]['total_questions'])*N
                eval_results[prompt_type]['all_correct_%']=(eval_results[prompt_type]['all_correct']/eval_results[prompt_type]['total_questions'])*N
                eval_results[prompt_type]['all_DK_%']=(eval_results[prompt_type]['all_DK']/eval_results[prompt_type]['total_questions'])*N
                eval_results[prompt_type]['correct_and_dk_%']=(eval_results[prompt_type]['correct_and_dk']/eval_results[prompt_type]['total_questions'])*N
                eval_results[prompt_type]['total_correct_%']=eval_results[prompt_type]['total_correct']/eval_results[prompt_type]['total_questions']
                eval_results[prompt_type]['total_DK_%']=eval_results[prompt_type]['total_DK']/eval_results[prompt_type]['total_questions']

                # F1
                eval_results[prompt_type]['f1']=(eval_results[prompt_type]['f1']/eval_results[prompt_type]['total_questions'])

                # EXACT MATCH
                eval_results[prompt_type]['exact_match_atleastone_correct_%']=(eval_results[prompt_type]['exact_match_atleastone_correct']/eval_results[prompt_type]['total_questions'])*N
                eval_results[prompt_type]['exact_match_all_correct_%']=(eval_results[prompt_type]['exact_match_all_correct']/eval_results[prompt_type]['total_questions'])*N
                eval_results[prompt_type]['exact_match_total_correct_%']=eval_results[prompt_type]['exact_match_total_correct']/eval_results[prompt_type]['total_questions']

                # COMBINED 
                combined_results[threshold][prompt_type]['total_correct_%'] = eval_results[prompt_type]['total_correct_%']
                combined_results[threshold][prompt_type]['atleastone_correct_%'] = eval_results[prompt_type]['atleastone_correct_%']
                combined_results[threshold][prompt_type]['f1'] =  eval_results[prompt_type]['f1']
                combined_results[threshold][prompt_type]['exact_match_total_correct_%'] = eval_results[prompt_type]['exact_match_total_correct_%']
                combined_results[threshold][prompt_type]['exact_match_atleastone_correct_%'] = eval_results[prompt_type]['exact_match_atleastone_correct_%']


            if threshold==optimal_t: is_optimal_t = True
            else: is_optimal_t=False
            evalresults_path = '{}/triviaQA_eval_t_{:.04f}_optimalt_{}.json'.format(path, threshold, is_optimal_t)

            # Eval: check for substring match, check standard eval for trivia qa for free text
            out_file = open(evalresults_path, "w")
            json.dump(eval_results, out_file, indent = 6)

            print("Saving results in {}".format(evalresults_path))
            # incorrect_results['dem_incorrect_noprompt_correct'] = list(set(incorrect_results['dem_questions']).difference(set(incorrect_results[''])))
            # incorrect_results['repub_incorrect_noprompt_correct'] = list(set(incorrect_results['repub_questions']).difference(set(incorrect_results['noprompt_questions'])))

            # incorrect_results['dem_correct_noprompt_incorrect'] = list(set(incorrect_results['noprompt_questions']).difference(set(incorrect_results['dem_questions'])))
            # incorrect_results['repub_correct_noprompt_incorrect'] = list(set(incorrect_results['noprompt_questions']).difference(set(incorrect_results['repub_questions'])))

            # incorrect_results_path = '/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/summarization/00{}/{}{}/incorrect.json'.format(args.model_num, args.prompt_strength, prompt_stength_addon)
            # out_file = open(incorrect_results_path, "w")
            # json.dump(incorrect_results, out_file, indent = 6)

        combinedresults_path = '{}/triviaQA_eval_combined.json'.format(path)
        print("Saving results in {}".format(combinedresults_path))
        out_file = open(combinedresults_path, "w")
        json.dump(combined_results, out_file, indent = 6)
    
    return optimal_t