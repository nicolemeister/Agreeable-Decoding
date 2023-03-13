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
from decoding_summarization import summarization_decoding
from eval_SUM import eval
import pandas as pd
import get_questions
import get_prompts
from triviaQA_eval import triviaQAeval
from subjective_information_tradeoff import subj_inform_eval

sys.path.append('helm')
sys.path.append('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/helm/src')
from src.helm.common.authentication import Authentication
from src.helm.proxy.services.remote_service import RemoteService, RemoteServiceError
from src.helm.common.request import Request

parser = argparse.ArgumentParser()

parser.add_argument('--model_output_dir', default='/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results')
parser.add_argument('--question_type', default='polarizingqs', type=str) # polarizingqs or neutralqs
parser.add_argument('--prompt_strength', default='strong', type=str) # normal or strong 
parser.add_argument('--model_num', default='2', type=int) # text-davinici 002 or 003
parser.add_argument('--temp', default='0.5', type=float)
parser.add_argument('--N', default=3, type=int)
parser.add_argument('--mini', default=False, type=bool) # trivia qa mini
parser.add_argument('--test_thresholds', default=True, type=bool) # test all thresholds
parser.add_argument('--enforce_certainty', default=False, type=bool) # enforcing certainty does not increase accuracy, depreciated 
parser.add_argument('--personas', help='delimited list input', type=str)
parser.add_argument('--model_type', default="text-davinci-00", type=str) # could be chatgpt
parser.add_argument('--do_triviaQA_eval', default=False, type=bool)

args = parser.parse_args()

# # Access account and show my current quotas and usages
# account = service.get_account(auth)
# print(account.usages)

prompts = get_prompts.prompts(args)
questions = get_questions.questions(args)

'''
RUN SUMMARIZATION DECODING
Run summarization_decoding() and get a results_gen that maps question to another dict with 
{persona} outputs, {persona}_logprobs, noprompt, noprompt_logprobs, SUM, SUM_variations
Stored in N_3.json
'''

ALL_RESULTS = []
# save the data 
if args.enforce_certainty: path = "{}/summarization/00{}/{}_enforce_certainty/{}/temp_{}/{}/".format(args.model_output_dir, args.model_num, args.prompt_strength, args.question_type, args.temp, ''.join(list(prompts.keys())))
else: path = "{}/summarization/00{}/{}/{}/temp_{}/{}/".format(args.model_output_dir, args.model_num, args.prompt_strength, args.question_type, args.temp, ''.join(list(prompts.keys())))
if not os.path.exists(path): os.makedirs(path)

# # ****** run noprompt only once 
# if not os.path.exists(path+'noprompt'):


BATCH_SIZE = 1
for i in range(BATCH_SIZE):
    print(i)
    result_gen = summarization_decoding(args, questions[int(i*len(questions)/BATCH_SIZE): int((i+1)*len(questions)/BATCH_SIZE)], prompts, N=args.N, enforce_certainty=args.enforce_certainty)
    out_file = open("{}/N_{}_i_{}.json".format(path, args.N, i), "w")
    json.dump(result_gen, out_file, indent = 6)
    ALL_RESULTS.append(result_gen.copy())

# merge all results from batch in to one dict
result_gen = {}
for d in ALL_RESULTS:
    result_gen.update(d)

# save the data 
if not os.path.exists(path): os.makedirs(path)
out_file = open("{}/N_{}.json".format(path, args.N), "w")
json.dump(result_gen, out_file, indent = 6)



# # merge in the no po
# if not os.path.exists(path+'noprompt'):



# read the data 
# read in the data polarizingqs_N_3.json
with open("{}/N_{}.json".format(path, args.N), 'r') as f:
    result_gen = json.load(f)


'''
EVAL: Evaluate the results of the summarization decoding with agreement metric
Store in score_N_3.json
'''

# result_eval = eval(args, questions, prompts, result_gen=result_gen, enforce_certainty=args.enforce_certainty)
# if args.enforce_certainty: 
#     for question in questions: 
#         result_eval[question[:-8] + ' If you do not know, please give your best guess.\nFriend:']['data'] = result_eval[question[:-8] + ' If you do not know, please give your best guess.\nFriend:']['data'].tolist()
# else: 
#     for question in questions: 
#         if not isinstance(result_eval[question]['data'], list): result_eval[question]['data'] = result_eval[question]['data'].tolist() # convert to json serializable
    

# with open("{}/score_N_3.json".format(path), "w") as outfile:
#     json.dump(result_eval, outfile, indent=6)


'''
EVAL (subset of) TRIVIAQA with the prompts 
- substring match, f1, and exact match scores for triviaQA
- Create plot of ROC curve for thresholding
- Create plot of trivia QA questions and polarizing questions with the optimal threshold highlighted 
- Create plot of triviaQA accuracy (total correct/num_q * n)
'''
# optimal_t=triviaQAeval(args, prompts, path)

'''
SUBJECTIVE INFORMATIVE EVALUATION: 
'''
# # generate the data
subj_inform_eval(args, path, prompts, result_gen)
