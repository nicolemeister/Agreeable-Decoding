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
from thresholding import optimal_threshold
import matplotlib.pyplot as plt


sys.path.append('helm')
sys.path.append('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/helm/src')
from src.helm.common.authentication import Authentication
from src.helm.proxy.services.remote_service import RemoteService, RemoteServiceError
from src.helm.common.request import Request

def add_ex_to_prompt(prompt, questions, answers):
    for idx, i in enumerate(range(7, 7+len(questions))):
        prompt +=  "## Example {}: ".format(str(i)) + '\n' + '\n'+\
                   "### Question {}: ".format(str(i)) + '\n' +questions[idx] + '\n' + '\n' + \
                   "### Answer {}: ".format(str(i)) + '\n' +answers[idx] + '\n' + '\n'

    prompt+="## Correct answers for examples 7-{}:".format(str(7+len(questions)-1))+ '\n' + '\n' + "### Likert score for example 7:"
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

def get_data(answers, polarizing_qs, nonpolarizing_qs, prompts, labels_gen, label_to_N):
    '''
    labels_gen = list(prompts.keys()) + ['noprompt', 'SUM', 'SUM_noquestion', 'SUM_tatsu']
    label_to_N = {'noprompt': 3, 'SUM': 1, 'SUM_noquestion': 1, 'SUM_tatsu': 1}
    for prompt in list(prompts.keys()):
        label_to_N[prompt]=3
    '''
    subjective_data = defaultdict(list)
    informative_data = defaultdict(list)

    noprompt_idx = len(prompts)*3

    for question in polarizing_qs: 
        answers_informative = answers[question]['answers_informative']
        answers_subjective = answers[question]['answers_subjective']
        for i, label in enumerate(labels_gen):
            # personas + no prompt have 3 ans
            if label in prompts or label=='noprompt':
                informative_data[label]+=list(map(int, answers_informative[(i*3):(i*3)+3]))
                subjective_data[label]+=list(map(int, answers_subjective[(i*3):(i*3)+3]))
            # summary stuff only has 1 ans
            else:
                informative_data[label].append(int(answers_informative[i*3]))
                subjective_data[label].append(int(answers_subjective[i*3]))

    # the polarizing question is below the threshold so thus we get the no prompt response
    for question in nonpolarizing_qs: 
        answers_informative = answers[question]['answers_informative']
        answers_subjective = answers[question]['answers_subjective']

        for i, label in enumerate(labels_gen):
            # personas + no prompt have 3 ans
            if label in prompts or label=='noprompt':
                informative_data[label]+=list(map(int, answers_informative[noprompt_idx:noprompt_idx+3]))
                subjective_data[label]+=list(map(int, answers_subjective[noprompt_idx:noprompt_idx+3]))
            # summary stuff only has 1 ans
            else:
                informative_data[label].append(int(answers_informative[noprompt_idx]))
                subjective_data[label].append(int(answers_subjective[noprompt_idx]))

    return informative_data, subjective_data



def get_mean_subj_info(answers, args, path, prompts, labels_gen, label_to_N):

    if args.model_num==3:
        path_triviaQA = '{}/summarization/00{}/{}/triviaQA/temp_{}/'.format(args.model_output_dir, '2', args.prompt_strength, args.temp)
        path_triviaQAmini = '{}/summarization/00{}/{}/triviaQAmini/temp_{}/'.format(args.model_output_dir, '2', args.prompt_strength, args.temp)
    
    else: 
        path_triviaQA = '{}/summarization/00{}/{}/triviaQA/temp_{}/'.format(args.model_output_dir, args.model_num, args.prompt_strength, args.temp)
        path_triviaQAmini = '{}/summarization/00{}/{}/triviaQAmini/temp_{}/'.format(args.model_output_dir, args.model_num, args.prompt_strength, args.temp)
    path_polarizingqs = path

    triviaQAmini_scores_path=path_triviaQAmini+"score_N_{}.json".format(args.N)
    triviaQA_scores_path=path_triviaQA+"score_N_{}.json".format(args.N)
    polarizingqs_scores_path=path_polarizingqs+"score_N_{}.json".format(args.N)

    optimal_t, trivia_offdiaggap, polarizing_offdiaggap, polarizing_offdiaggap_dict = optimal_threshold(polarizingqs_scores_path, triviaQAmini_scores_path, triviaQA_scores_path)
    thresholds = [-0.05]
    for i in np.linspace(0, 75, 76):
        thresholds.append(np.percentile(polarizing_offdiaggap, i))

    t_to_SI = defaultdict(dict)
    for t in thresholds:
        # based on threshold, get list of polarizing qs
        polarizing_qs, nonpolarizing_qs = [], []
        for q in polarizing_offdiaggap_dict.keys():
            agreement_score = polarizing_offdiaggap_dict[q]
            if agreement_score >= t: 
                polarizing_qs.append(q)
            else: 
                nonpolarizing_qs.append(q)

        # map threshold to mean (subjective rating, info rating)
        informative_data, subjective_data = get_data(answers, polarizing_qs, nonpolarizing_qs, prompts, labels_gen, label_to_N)
        for label in labels_gen:
            t_to_SI[t][label] = [np.mean(informative_data[label]), np.mean(subjective_data[label])]

    return t_to_SI, optimal_t


def subj_inform_eval(args, path, prompts, result_gen):
    # labels_gen = list(prompts.keys()) + ['noprompt', 'SUM', 'SUM_noquestion', 'SUM_tatsu']
    # label_to_N = {'noprompt': 3, 'SUM': 1, 'SUM_noquestion': 1, 'SUM_tatsu': 1}

    labels_gen = list(prompts.keys()) + ['noprompt', 'SUM']
    label_to_N = {'noprompt': 3, 'SUM': 1}


    
    for prompt in list(prompts.keys()):
        label_to_N[prompt]=3

    '''
    
    # read in the informative instruction prompt 
    with open('{}/../helm/prompt_informative.txt'.format(args.model_output_dir)) as f:
        lines = f.readlines()
    informative_prompt_instructions = ''.join(lines)

    # read in the subjective instruction prompt 
    with open('{}/../helm/prompt_subjective.txt'.format(args.model_output_dir)) as f:
        lines = f.readlines()
    subjective_prompt_instructions = ''.join(lines)

    # extract questions and their answers 
    # read in the data polarizingqs_N_3.json
    # with open("{}/N_{}.json".format(path, args.N), 'r') as f:
    #     result_gen = json.load(f)

    questions = list(result_gen.keys())

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
        informative_output, _ = get_text_logprob(args, final_informative_prompt, max_tokens=700)
        informative_output = parse(informative_output) # parse output
        results[question]['answers_informative'] = informative_output
        print(informative_output)

        # SUBJECTIVE
        final_subjective_prompt = add_ex_to_prompt(prompt=subjective_prompt_instructions, questions=[question for i in range(len(answers))], answers=answers)
        subjective_output, _ = get_text_logprob(args, final_subjective_prompt, max_tokens=700)
        subjective_output = parse(subjective_output)
        results[question]['answers_subjective'] = subjective_output
        print(subjective_output)


    args.model_num = prev_arg_model_num
    # save the data 
    if not os.path.exists(path): os.makedirs(path)
    out_file = open("{}/subjective_informative.json".format(path, args.N), "w")
    json.dump(results, out_file, indent = 6)
    '''

    with open("{}/subjective_informative.json".format(path), 'r') as f:
        results = json.load(f)

    # # combine large and small
    # small_002.update(large_002)

    t_to_SI, optimal_t = get_mean_subj_info(results, args, path, prompts, labels_gen, label_to_N)

    # plot
    fig3 = plt.figure("Figure 3")
    for label in t_to_SI[-0.05].keys():
        x, y = [], []
        for t in t_to_SI.keys():
            if t_to_SI[t]:
                x.append(t_to_SI[t][label][0])
                y.append(t_to_SI[t][label][1])
        plt.scatter(x,y, s=4, label=label)
    plt.scatter(x=3.95, y=1.05, label='optimal system')
    plt.legend()
    plt.xlabel('Informative')
    plt.ylabel('Subjective')
    plt.title("{}{}".format(args.model_type, args.model_num))
    plt.xlim([1, 4])
    plt.ylim([1, 4])
    plt.savefig("{}/subjective_informative_plot_all.jpg".format(path), dpi=1200)
    plt.show()

    # plot optimal threshold values
    fig4 = plt.figure("Figure 4")
    for label in t_to_SI[-0.05].keys():
        plt.scatter(x=t_to_SI[optimal_t][label][0], y=t_to_SI[optimal_t][label][1], s=6, label="optimal " +label)
    plt.scatter(x=3.95, y=1.05, label='optimal system')
    plt.legend(loc='lower left')
    plt.xlabel('Informative')
    plt.ylabel('Subjective')
    plt.title("{}{}".format(args.model_type, args.model_num))
    plt.xlim([1, 4])
    plt.ylim([1, 4])
    print("{}/subjective_informative_plot_optimal.jpg".format(path))
    plt.savefig("{}/subjective_informative_plot_optimal.jpg".format(path), dpi=1200)

    return