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


sys.path.append('helm')
sys.path.append('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/helm/src')
from src.helm.common.authentication import Authentication
from src.helm.proxy.services.remote_service import RemoteService, RemoteServiceError
from src.helm.common.request import Request


auth = Authentication(api_key='CH8doliaueEL6vDXuwaqOmxa3Hld4dT2')
service = RemoteService("https://crfm-models.stanford.edu")


def format_q(question):
    return "You: " + question + "\nFriend:"

# given a prompt, get the text and logprobs 
def get_text_logprob(args, input, max_tokens):
    breakpoint()
    prompt = input

    if args.model_type=='chatgpt':
        pass
    else: 
        request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt, max_tokens=max_tokens, echo_prompt=False, temperature=args.temp, random=str(random.randint(1,1000)))
        try:
            request_result = service.make_request(auth, request)
        except RemoteServiceError:
            print('Error making request')
            request_result = service.make_request(auth, request)
        return_tokens = request_result.completions[0].tokens
        output = ''.join(np.array([t.text for t in return_tokens]))
        logprobs=[t.logprob for t in return_tokens]
    return output, logprobs

# given a list of opinions, summarize them
def summarize(args, question, opinions):
    prompt = "Summarize these answers to the following question in a way that includes the opinions of both sides. " + question[5:-8] + '\n'
    for opinion in opinions:
        prompt+="\n-"+opinion
    prompt+='\nSummary:'
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt, max_tokens=150, echo_prompt=False, temperature=0.0, random=str(random.randint(1,1000)))
    try:
        request_result = service.make_request(auth, request)
    except RemoteServiceError:
        print('Error making request')
    return_tokens = request_result.completions[0].tokens
    output = ''.join(np.array([t.text for t in return_tokens]))
    logprobs=[t.logprob for t in return_tokens]
    return output, logprobs

def summarize_noquestion(args, opinions):
    prompt = "Summarize these answers to a question.\n"
    for opinion in opinions:
        prompt+="\n-"+opinion
    prompt+='\nSummary:'
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt, max_tokens=150, echo_prompt=False, temperature=0.0, random=str(random.randint(1,1000)))
    try:
        request_result = service.make_request(auth, request)
    except RemoteServiceError:
        print('Error making request')
    return_tokens = request_result.completions[0].tokens
    output = ''.join(np.array([t.text for t in return_tokens]))
    logprobs=[t.logprob for t in return_tokens]
    return output, logprobs

def summarize_tatsu(args, question, opinions):
    prompt = 'Here are some possible ways to respond to the question "' + question[5:-8] + '"\n'
    for opinion in opinions:
        prompt+="\n- "+opinion
    prompt+='\n Please combine these responses into one summary response.'
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt, max_tokens=150, echo_prompt=False, temperature=0.0, random=str(random.randint(1,1000)))
    try:
        request_result = service.make_request(auth, request)
    except RemoteServiceError:
        print('Error making request')
    return_tokens = request_result.completions[0].tokens
    output = ''.join(np.array([t.text for t in return_tokens]))
    logprobs=[t.logprob for t in return_tokens]
    return output, logprobs

# input: list of questions, prompt_dem, prompt_repub, N
# output: result 
# what the function does: compute agreeable decoding, 
# if enforce_certainty, append prompt "Make sure you provide an answer or "do not say i dont' know, please provide your best guess"
def summarization_decoding(args, questions, prompts, N, enforce_certainty = False):
    result = {}
    # number of responses from each persona
    if args.temp<=0: N=1
    for question in questions:
        if enforce_certainty: question= question[:-8] + ' If you do not know, please give your best guess.\nFriend:'
        result[question]=defaultdict(list)
        for k in range(N):
            # get N arguments from persona
            # ADD NO PROMPT TO PERSONAS
            prompts['noprompt'] = ''
            for persona in prompts:
                S, logprobs = get_text_logprob(args, prompts[persona]+question, max_tokens=100)
                result[question][persona].append(S)
                result[question]["{}_logprobs".format(persona)].append(logprobs)
            
        # treat as summarization task to summarize the num_personas * N opinions
        
        # SUMMARY 1: summary prompt includes question
        ans_to_summarize = []
        for persona in prompts:
            ans_to_summarize+=result[question][persona]
        summarized_output, logprobs = summarize(args, question, ans_to_summarize)
        print("Agreeable decoding: ", summarized_output)
        result[question]['SUM'].append(summarized_output) # store the summarized response 
        result[question]['SUM_logprobs'].append(logprobs)

        # # SUMMARY 2: summary prompt does not include question]
        # summarized_output, logprobs = summarize_noquestion(args, ans_to_summarize)
        # print("Agreeable decoding without question: ", summarized_output)
        # result[question]['SUM_noquestion'].append(summarized_output) # store the summarized response 
        # result[question]['SUM_noquestion_logprobs'].append(logprobs)

        # # SUMMARY 3: summary prompt based on tatsu's rec prompt 
        # summarized_output, logprobs = summarize_tatsu(args, question, ans_to_summarize)
        # print("Agreeable decoding with tatsu's prompt: ", summarized_output)
        # result[question]['SUM_tatsu'].append(summarized_output) # store the summarized response 
        # result[question]['SUM_tatsu_logprobs'].append(logprobs)

    return result


