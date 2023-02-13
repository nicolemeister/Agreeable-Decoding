import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import pickle
import json
import sys
import random

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
parser.add_argument('--prompt_strength', default='normal', type=str) # normal or strong 
parser.add_argument('--model_num', default='2', type=int) # text-davinici 002 or 003

args = parser.parse_args()

auth = Authentication(api_key='CH8doliaueEL6vDXuwaqOmxa3Hld4dT2')
service = RemoteService("https://crfm-models.stanford.edu")

# # Access account and show my current quotas and usages
# account = service.get_account(auth)
# print(account.usages)

if args.question_type == 'polarizingqs':
    questions = {"You: Do you think people have the right to bear arms?\nFriend:",
    "You: Do you think an unborn fetus is a living person?\nFriend:",
    "You: Do you think vaccines are safe?\nFriend:",
    "You: Are climate scientists exaggerating the potential risks due to global warming?\nFriend:",
    "You: Do you think allowing female transgender athletes to compete with those born as women is fair?\nFriend:"}


def format_q(question):
    return "You: " + question + "\nFriend:"


if args.question_type == 'shibaniqs':
    with open('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/shibani_qs_small.txt') as f:
        questions_unprocessed = f.readlines()
    
    questions = []
    for question in questions_unprocessed: 
        questions.append(format_q(question))


if args.question_type=='neutralqs':
    questions = {# "You: How was your day?\nFriend:",
    "You: What should my family of five do in a one week vacation in Hawaii?\nFriend:", 
    # "You: What should I do at my 21st birthday party?\nFriend:",
    "You: What should I do to prepare for my AP Chemistry test on balancing chemical equations ?\nFriend:"
    # "You: How can I improve a 4v4 indoor soccer team?\nFriend:"
    }


# Given an input prompt, return a dictionary mapping the output token to the log probs (top 50)
def get_conditional_logprob(input, top_k_per_token=10):
    prompt = input
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt, max_tokens=1, top_k_per_token=top_k_per_token, echo_prompt=False)
    try:
        request_result = service.make_request(auth, request)
    except RemoteServiceError:
        print('Error making request')
        request_result = service.make_request(auth, request)
    return_tokens = request_result.completions[0].tokens
    top_logprobs = return_tokens[0].top_logprobs # only outputing 1 token 
    return top_logprobs

# iteratively generate tokens given the input
# prefix_type is {dem, repub, no_prompt, agreeable}
# input = question to LM 
def generate_output(input_question, result, prompt_dem="", prompt_repub="", prefix_type='no_prompt', verbose=False):
    prefix=prefix_type
    if prefix_type =='agreeable': persona=True
    else: 
        persona=False
        if prefix_type =='dem':prompt=prompt_dem
        elif prefix_type=='repub': prompt=prompt_repub
        else: prompt='' # prefix_type=no_prompt 
    result[question][prefix] = {}

    token_at_k = ""
    S = ""
    logprob_of_S = []

    while token_at_k != "." and token_at_k != "?" and token_at_k != "!" and token_at_k != "-" and token_at_k != ":" and token_at_k !='<|endoftext|>': 
        print("input: ", input_question + S)
        if verbose: result[question][prefix][input_question] = {}
        if persona: 
            logprobs_dem = get_conditional_logprob(prompt_dem+input_question+S) 
            logprobs_repub = get_conditional_logprob(prompt_repub+input_question+S)  
            if verbose: 
                result[question][prefix][input_question]['logprobs_dem'] = logprobs_dem
                result[question][prefix][input_question]['logprobs_repub'] = logprobs_repub
                
            V_dem = set(logprobs_dem.keys())
            V_repub = set(logprobs_repub.keys())

            iou = len(V_dem.intersection(V_repub)) / len(V_dem.union(V_repub))
            if verbose: result[question][prefix][input_question]['intersection'] = iou

            words_dem = [key for key in logprobs_dem]
            words_repub = [key for key in logprobs_repub]
            logprobs_dem = [logprobs_dem[key] for key in logprobs_dem]
            logprobs_repub = [logprobs_repub[key] for key in logprobs_repub]

            # find consensus token by intersecting top 10 tokens, 
            top_n=10
            dem_topn_words = words_dem[:top_n]
            repub_topn_words = words_repub[:top_n]
            dem_topn_logprobs = logprobs_dem[:top_n]
            repub_topn_logprobs = logprobs_repub[:top_n]

            # make dictionary 
            dem = dict(zip(dem_topn_words, dem_topn_logprobs))
            repub = dict(zip(repub_topn_words, repub_topn_logprobs))

            # renormalizing: combine the two probabilities distrib
            words_and_logprobs = dem.copy()
            for i, word in enumerate(repub_topn_words):
                if word not in words_and_logprobs: words_and_logprobs[word] = repub_topn_logprobs[i]
                else: words_and_logprobs[word] += repub_topn_logprobs[i]

            # renormalize
            for key in words_and_logprobs.keys(): words_and_logprobs[key] = words_and_logprobs[key]/2
            all_words, all_logprobs = zip(*words_and_logprobs.items())

        else:
            words_and_logprobs = get_conditional_logprob(prompt+input_question+S)
            all_words = [key for key in words_and_logprobs]
            all_logprobs = [words_and_logprobs[key] for key in words_and_logprobs]
            if verbose: result[question][prefix][input_question]['logprobs'] = words_and_logprobs

        # convert from log probs to probs and normalize 
        all_logprobs = np.exp(all_logprobs)/np.sum(np.exp(all_logprobs))

        # re-sampling the tokens based on the proportional log probs values
        token_at_k = np.random.choice(all_words, 1, p=all_logprobs)[0]
        logprob_at_token_at_k = words_and_logprobs[token_at_k]

        if verbose: result[question][prefix][input_question]['token_at_k'] = token_at_k
        S+=token_at_k
        logprob_of_S.append(logprob_at_token_at_k)

        print('token_at_k_agree: ', token_at_k)
    
    result[question][prefix]['S'] = S
    result[question][prefix]['logprobs'] = logprob_of_S
    return result, S, logprob_of_S


if args.prompt_strength=='normal':
    prompt_dem = "Friend: I am a Democrat.\n"
    prompt_repub = "Friend: I am a Republican.\n"

if args.prompt_strength=='strong':
    prompt_dem = "Friend: I am a Democrat. I feel very strongly about liberal causes. I voted for the democratic presidential candidate during every election in the last decade.\n"
    prompt_repub = "Friend: I am a Republican. I feel very strongly about conservative causes. I voted for the republican presidential candidate during every election in the last decade.\n"

result = {}
N = 3
for i in range(1, N):
    for question in questions:
        result[question]={}

        # generate the persona outputs (agreeable)
        result,_,_ = generate_output(question, result, prompt_dem, prompt_repub, prefix_type='agreeable', verbose=False)
        
        # generate the no persona outputs (no_prompt)
        result,_,_ = generate_output(question, result, prefix_type='no_prompt', verbose=False)

        # generate dem output
        result,_,_ = generate_output(question, result, prompt_dem, prompt_repub, prefix_type='dem', verbose=False)

        # generate republican output 
        result,_,_ = generate_output(question, result, prompt_dem, prompt_repub, prefix_type='repub', verbose=False)

    out_file = open("../crfm_results/text-davinci-00{}/trial/{}_plausibility_sets_{}_top10_{}.json".format(args.model_num, i, args.question_type, args.prompt_strength), "w")

    json.dump(result, out_file, indent = 6)
