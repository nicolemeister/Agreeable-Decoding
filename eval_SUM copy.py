import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import pickle
import json
import sys
import random
from collections import defaultdict
from itertools import combinations


sys.path.append('helm')
sys.path.append('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/helm/src')
from src.helm.common.authentication import Authentication
from src.helm.proxy.services.remote_service import RemoteService, RemoteServiceError
from src.helm.common.request import Request

auth = Authentication(api_key='CH8doliaueEL6vDXuwaqOmxa3Hld4dT2')
service = RemoteService("https://crfm-models.stanford.edu")

# Access account and show my current quotas and usages
account = service.get_account(auth)
print(account.usages)

# Given an input prompt, return a dictionary mapping the output token to the log probs (top 50)
def get_logprob_ofinput(args, input):
    prompt, question, output = input
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt+question+output, max_tokens=0, echo_prompt=True, temperature=0.0) 
    try:
        request_result = service.make_request(auth, request)
    except RemoteServiceError:
        print('Error making request')

    return_tokens = request_result.completions[0].tokens
    cum_text_lens = np.cumsum([len(t.text) for t in return_tokens])
    output_idx_start = np.argmax(cum_text_lens > len(prompt+question))
    logprobs = np.array([t.logprob for t in return_tokens])
    # text = np.array([t.text for t in return_tokens])
    return logprobs[output_idx_start:]


# input: data/results, questions, prompts
# output: data/results
# what the function does: compute off diagonal gap for question in the results file
def eval(args, questions, prompts, result_gen, enforce_certainty=False):
    label_to_idx_eval, label_to_idx_gen, label_to_N = {}, {}, {}

    data = result_gen
    prompts['noprompt']= ""

    for i, persona in enumerate(list(prompts.keys())):
        label_to_idx_eval[persona] = i
        label_to_idx_gen[persona] = i
        label_to_N[persona] = args.N

    label_to_idx_gen['SUM']= len(prompts)
    label_to_N['SUM']=1

    result=defaultdict(dict)

    for question in questions:
        if enforce_certainty: question= question[:-8] + ' If you do not know, please give your best guess.\nFriend:'
        result[question]['data'] = np.zeros((len(prompts)+1,len(prompts))) # if two personas it's 4x3 (cols: D, R, noprompt, sum, rows: D, R, noprompt)
    
    # # read in the data polarizingqs_N_3.json
    # with open("{}/{}/00{}/{}/temp_{}/{}_N_3.json".format(args.model_output_dir, args.input_data, args.model_num, args.prompt_strength, args.temp, args.question_type), 'r') as f:
    #     data = json.load(f)


    for question in questions:
        if enforce_certainty: question= question[:-8] + ' If you do not know, please give your best guess.\nFriend:'
        # [gen] for each persona (dem, repub, no prompt, SUM), grab the output S from P_Persona
        for key in label_to_idx_gen.keys():
            N=label_to_N[key]
            for i in range(N):
                S = data[question][key][i]
                logprobs_gen = np.array(data[question][key+'_logprobs'][i])
                # [eval] for each model persona (dem, repub, noprompt) 
                for eval_type in list(prompts.keys()):
                    # run the output S through the model
                    prompt = prompts[eval_type]
                    output = S
                    logprobs_eval = get_logprob_ofinput(args, input=(prompt, question, output))
                    # sum up the log probs and divide by len(S) (MEAN) and the exponentiate
                    if len(logprobs_eval) != len(logprobs_gen): #/n/n/n
                        print(S)
                        try: cross_entropy = -1*np.multiply(np.exp(logprobs_gen[3:]), logprobs_eval[2:]).mean()
                        except: cross_entropy=0 
                    else:
                        cross_entropy = -1*np.multiply(np.exp(logprobs_gen), logprobs_eval).mean()
                    
                    result[question]['data'][label_to_idx_gen[key]][label_to_idx_eval[eval_type]] += ((1/N) * cross_entropy)

    result['avg']={}
    result['avg']['data'] = np.zeros((len(prompts)+1,len(prompts)))
    result['avg']['off_diagonal_gap'] = 0



    # off diagonal gap: off diagonal - diagonal (H(D, R) + H(R, D) - H(D) - H(R))
    for question in questions:
        if enforce_certainty: question= question[:-8] + ' If you do not know, please give your best guess.\nFriend:'
        # prompts-1 because right now prompts includes noprompt
        off_diag_gaps = []
        for i,j in combinations(np.arange(len(prompts)-1), 2):
            off_diag_gaps.append(result[question]['data'][i][j] + \
                                        result[question]['data'][j][i] - \
                                        result[question]['data'][i][i] - \
                                        result[question]['data'][j][j])

        result[question]['off_diagonal_gap'] = np.max(np.array(off_diag_gaps))
        result['avg']['data']+= result[question]['data']
        result['avg']['off_diagonal_gap'] += result[question]['off_diagonal_gap'] 

    result['avg']['data']= result['avg']['data'] / len(questions)
    result['avg']['off_diagonal_gap']= result['avg']['off_diagonal_gap'] / len(questions)
    result['avg']['data']=result['avg']['data'].tolist()

    return result
