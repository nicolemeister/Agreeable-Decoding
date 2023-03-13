import argparse
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import multiprocessing as mp
import os
import os.path
from os import path
from collections import defaultdict

def format_q(question):
    return "You: " + question + "\nFriend:"

def questions(args):
    if args.question_type == 'polarizingqs_1':
        questions = ["You: Do you think people have the right to bear arms?\nFriend:"]

    if args.question_type == 'polarizingqs_3':
        questions = ["You: Do you think people have the right to bear arms?\nFriend:", 
                    "You: Do you think an unborn fetus is a living person?\nFriend:",
                    "You: Do you think vaccines are safe?\nFriend:",]


    if args.question_type == 'polarizingqs':
        questions = ["You: Do you think people have the right to bear arms?\nFriend:",
        "You: Do you think an unborn fetus is a living person?\nFriend:",
        "You: Do you think vaccines are safe?\nFriend:",
        "You: Are climate scientists exaggerating the potential risks due to global warming?\nFriend:",
        "You: Do you think allowing female transgender athletes to compete with those born as women is fair?\nFriend:"
        ]

    if args.question_type == 'polarizingqs_large':
        with open('{}/shibani_qs_polarizing.txt'.format(args.model_output_dir)) as f:
            questions_unprocessed = f.readlines()
        
        questions = []
        for question in questions_unprocessed: 
            questions.append(format_q(question))

        
    if args.question_type == 'shibaniqs_25':
        with open('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/shibani_qs_25.txt') as f:
            questions_unprocessed = [line.rstrip() for line in f]

        questions = []
        for question in questions_unprocessed: 
            questions.append(format_q(question))

    if args.question_type == 'triviaQAmini_subset':
        questions = ["You: Miami Beach in Florida borders which ocean?\nFriend:", 
                    "You: What was the occupation of Lovely Rita according to the song by the Beatles?\nFriend:",
                    "You: Who was Poopdeck Pappys most famous son?\nFriend:",
                    "You: The Nazi regime was Germany's Third Reich; which was the first Reich?\nFriend:",
                    "You: At which English racecourse did two horses collapse and die in the parade ring due to electrocution, in February 2011?\nFriend:",
        ]


    if args.question_type == 'shibaniqs_small':
        with open('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/shibani_qs_small.txt') as f:
            questions_unprocessed = f.readlines()
        
        questions = []
        for question in questions_unprocessed: 
            questions.append(format_q(question))


    if args.question_type == 'triviaQAmini':
        df = pd.read_excel('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/triviaQA_subset.xlsx')
        
        questions = []
        for question in df['Question'][:100]: 
            questions.append(format_q(question))

    if args.question_type == 'triviaQA':
        df = pd.read_excel('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/triviaQA_subset.xlsx')
        
        questions = []
        for question in df['Question'][100:]: 
            questions.append(format_q(question))

    if args.question_type=='neutralqs':
        questions = {# "You: How was your day?\nFriend:",
        "You: What should my family of five do in a one week vacation in Hawaii?\nFriend:", 
        # "You: What should I do at my 21st birthday party?\nFriend:",
        "You: What should I do to prepare for my AP Chemistry test on balancing chemical equations ?\nFriend:"
        # "You: How can I improve a 4v4 indoor soccer team?\nFriend:"
        }
    return questions
