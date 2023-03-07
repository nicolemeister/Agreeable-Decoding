import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import pickle
import json
import sys
import random

path_prefix='/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/'

filenames = [
            'crfm_results/output_only/plausibility_sets_neutralqs_top10_normal.json',
            'crfm_results/output_only/plausibility_sets_polarizingqs_top10_normal.json',
            'crfm_results/output_only/plausibility_sets_polarizingqs_top10_strong.json',
            'crfm_results/text-davinci-002/0_plausibility_sets_polarizingqs_top10_normal.json',
            'crfm_results/text-davinci-002/0_plausibility_sets_polarizingqs_top10_strong.json',
            'crfm_results/text-davinci-002/1_plausibility_sets_polarizingqs_top10_normal.json',
            'crfm_results/text-davinci-002/2_plausibility_sets_polarizingqs_top10_normal.json',
            'crfm_results/text-davinci-002/3_plausibility_sets_polarizingqs_top10_normal.json',
            'crfm_results/text-davinci-002/4_plausibility_sets_polarizingqs_top10_normal.json']

types = ['dem', 'repub', 'no_prompt', 'agreeable']
for filename in filenames:
    print(filename)
    with open(path_prefix + filename, 'r') as f:
        data = json.load(f)
    questions = data.keys()

    for question in questions:
        # [gen] for each persona (dem, repub, no_prompt, agreeable), grab the output S from P_Persona
        for key in types:
            S = data[question][key]['S']
            newS = S[7:]
            # S_split = S.split('\n')[1:]
            # newS = ''.join(S_split)
            data[question][key]['S'] = newS
    
    out_file = open(path_prefix+filename, "w")
    json.dump(data, out_file, indent = 6)
  
    