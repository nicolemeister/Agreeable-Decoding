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
    questions = {# "You: Do you think people have the right to bear arms?\nFriend:",
    "You: Do you think an unborn fetus is a living person?\nFriend:",
    "You: Do you think vaccines are safe?\nFriend:",
    "You: Are climate scientists exaggerating the potential risks due to global warming?\nFriend:",
    "You: Do you think allowing female transgender athletes to compete with those born as women is fair?\nFriend:"
    }


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

# given a prompt, get the text and logprobs 
def get_text_logprob(input):
    prompt = input
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt, max_tokens=70, echo_prompt=False)
    try:
        request_result = service.make_request(auth, request)
    except RemoteServiceError:
        print('Error making request')
        request_result = service.make_request(auth, request)
    return_tokens = request_result.completions[0].tokens
    output = ''.join(np.array([t.text for t in return_tokens]))
    logprobs=[t.logprob for t in return_tokens]
    return output, logprobs

# Given an input prompt, return a dictionary mapping the output token to the log probs (top 50)
def get_conditional_logprob(input):
    prompt = input
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num), prompt=prompt, max_tokens=1, top_k_per_token=10, echo_prompt=False)
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
        elif prefix_type=='repub':prompt=prompt_repub
        else: prompt='' # prefix_type=no_prompt 
    result[question][prefix] = {}

    token_at_k = ""
    S = ""
    logprob_of_S = []

    if persona:
        while token_at_k != "." and token_at_k != "?" and token_at_k != "!" and token_at_k != "-" and token_at_k != ":" and token_at_k !='<|endoftext|>': 
            print("input: ", input_question + S)
            if verbose: result[question][prefix][input_question] = {}
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

            # convert from log probs to probs and normalize 
            all_logprobs = np.exp(all_logprobs)/np.sum(np.exp(all_logprobs))

            # re-sampling the tokens based on the proportional log probs values
            token_at_k = np.random.choice(all_words, 1, p=all_logprobs)[0]
            logprob_at_token_at_k = words_and_logprobs[token_at_k]

            if verbose: result[question][prefix][input_question]['token_at_k'] = token_at_k
            S+=token_at_k
            logprob_of_S.append(logprob_at_token_at_k)

            print('token_at_k_agree: ', token_at_k)
    
    else:
        S, logprob_of_S = get_text_logprob(prompt+input_question)
    
    result[question][prefix]['S'] = S
    result[question][prefix]['logprobs'] = logprob_of_S
    return result, S, logprob_of_S


if args.prompt_strength=='normal':
    prompt_dem = "Friend: I am a Democrat.\n"
    prompt_repub = "Friend: I am a Republican.\n"

if args.prompt_strength=='strong':
    prompt_dem = "Friend: I am a Democrat. I feel very strongly about liberal causes. I voted for the democratic presidential candidate during every election in the last decade.\n"
    prompt_repub = "Friend: I am a Republican. I feel very strongly about conservative causes. I voted for the republican presidential candidate during every election in the last decade.\n"

if args.prompt_strength=='LM_persona1':
    prompt_dem = "Friend: I believe in the ideal of a fair and just society, one in which everyone has an equal opportunity to succeed. I believe in the power of government to help create these opportunities and to level the playing field so that everyone can have a fair chance at achieving their dreams. I believe that government should be progressive and invested in ensuring that all people have access to quality education, health care, and other basic needs. Lastly, I am a firm believer in civil rights and civil liberties, and that all people should be treated with respect and dignity regardless of their race, ethnicity, gender, sexual orientation, or any other characteristic.\n"
    prompt_repub = "Friend: I am a responsible, hard-working individual who values the importance of personal responsibility, fiscal conservatism, and limited government. I believe in upholding traditional values and preserving the core principles of our nation's founding. I am committed to protecting individual liberties and promoting economic growth through free market solutions.\n"

if args.prompt_strength=='LM_convo1':
    prompt_dem = "You: Hey, how's it going?\nFriend: Pretty good, how about you?\nYou: Doing alright. So, what do you think about the current state of politics?\nFriend: To be honest, I'm not a fan. I think it's time for some real change. We need to move away from the status quo and start advocating for more progressive policies.\nYou: Absolutely! I completely agree. We need to fight for things like universal healthcare, higher taxes on the wealthy, and more investment in renewable energy sources.\nFriend: Absolutely! We need to make sure that everyone has access to the same opportunities and resources regardless of their background or income level. We also need to make sure that our environment is protected and that we're taking steps towards sustainability.\nYou: Definitely! It's so important that we take action now before it's too late. We also need to make sure that our government is held accountable and that our representatives are actually representing us and not just their own interests.\nFriend: Absolutely! We can't let them get away with anything they want without any consequences or oversight. We need to make sure they're doing what they were elected to do - represent us and our interests in a fair and equitable manner.\n"
    prompt_repub = "You: Did you hear about the latest news from the White House?\nFriend: Yeah, I did. It sounds like President Trump is continuing to make America great again!\nYou: Absolutely! I love seeing how he is fighting to protect traditional values and institutions. It's precisely why I voted for him.\nFriend: Me too. He's always willing to put America first and make decisions that will have a positive impact long-term, even if they are unpopular in the short term.\nYou: I couldn't agree more. And it's not just his foreign policy that I'm a fan of — he's also doing his best to create economic opportunity and remove burdensome regulations on businesses, both of which are putting Americans back to work at record levels.\nFriend: Yeah, I think Trump being elected has really been a shot in the arm for businesses throughout the country and has led to some massive job growth — both things we definitely need more of!\nYou: Agreed. And I'm especially thrilled with his Supreme Court appointments — Gorsuch and Kavanaugh have both been outstanding picks so far! His commitment to upholding our conservative values by appointing justices with conservative judicial views gives me comfort that our rights will be protected.\nFriend: Absolutely - we can't forget how important it is for us as conservatives to maintain majority on the court if we'm gonna push through important bills like tax reform and Obamacare repeal efforts, not mention defending religious freedoms for everyone no matter their beliefs!\n"

if args.prompt_strength=='LM_persona2':
    prompt_dem = "Friend: I am a Democrat with liberal values who believes in progressive policies that promote social and economic justice, respect for diversity, and the protection of civil liberties. I prioritize creating an equitable society where everyone has access to opportunity and can reach their full potential. I believe in the power of collective action and that we must work together to create a more just world.\n"
    prompt_repub = "Friend: I am a Republican with conservative values who believes in limited government, fiscal responsibility, individual liberty, and a strong national defense. I believe in personal responsibility and that it is the responsibility of individuals to take care of themselves and their families. I believe in free markets and that government should not interfere in business. I also believe in traditional values such as faith, family, and patriotism.\n"

if args.prompt_strength=='LM_convo2':
    prompt_dem = "You: Hey, what's up?\nFriend: Not much. Just trying to catch up on the news. What about you?\nYou: I'm trying to figure out how to get more involved in the political process.\nFriend: That's great! What have you been thinking about doing?\nYou: Well, I think I want to be more active in advocating for liberal causes and policies.\nFriend: That sounds like a great idea! Have you looked into any specific organizations or initiatives?\nYou: Yeah, I've been looking into some progressive groups that are working on issues like healthcare reform and climate change.\nFriend: Those are both really important issues. Have you thought about how you would like to get involved?\n"
    prompt_repub = "You: Hey there, what do you think about the current state of our nation?\nFriend: Well, it's not looking great. I'm a Republican and I have very conservative beliefs, so I'm not too happy with the way things are going. We need to get back to the core principles of our party and stand up for what we believe in.\nYou: Absolutely! We need to push for smaller government, lower taxes, and fiscal responsibility. We also need to make sure that we protect our freedoms and liberties.\nFriend: Exactly! And we need to make sure that our borders are secure and that illegal immigration is stopped. We must also ensure that our laws are enforced and that criminals are punished for their actions.\nYou: Yes, those are all important points. We should also make sure that we protect religious freedom and traditional values. Those are essential parts of our society and must be defended at all costs.\nFriend: Absolutely! We must also stand up for the Second Amendment rights of law-abiding citizens. It is essential that they have the right to defend themselves against any threat or danger they may face.\nYou: Definitely! And we must also push back against liberal policies like abortion on demand and same-sex marriage. These policies go against our beliefs as Republicans, so we must be vocal in opposing them whenever possible.\nFriend: Agreed! It's time for us to take a stand for what we believe in and fight for a better future for everyone in this country!"

if args.prompt_strength=='LM_convo3':
    prompt_dem = "You: Hey there, I'm so glad to see you! It's been a while since we've had the chance to catch up.\nFriend: It sure has! How have you been?\nYou: I'm doing great. Things have been busy but exciting. How about you?\nFriend: Same here. I've been trying to stay active in the political sphere, advocating for more liberal policies and raising awareness about the issues that matter most to me.\nYou: That's great! I've been doing a lot of the same lately. It's so important that we keep pushing for progressive change in our country, especially when it comes to social justice and equality.\nFriend: Absolutely! We need to make sure that everyone gets a fair shot in life, regardless of their race or background. We need to fight for things like better wages, healthcare access, and education reform so that everyone can live a comfortable life.\nYou: I couldn't agree more. We also need to make sure that we're protecting our environment and taking steps towards sustainability so that future generations can still enjoy the planet we live on today.\nFriend: That's right! We should be investing in renewable energy sources like solar and wind power as well as investing in green technology solutions like electric cars and smart grids so that we can reduce our reliance on fossil fuels and create a cleaner world for everyone.\nYou: Absolutely! We also need to make sure that our government is held accountable for its actions and that those in power are held responsible when they break the law or abuse their power. No one should be above the law, no matter who they are or what position they hold.\nFriend: Right on! We need more people like us who are willing to stand up for what's right and fight for real change in our country if we want to see any progress made in the near future."
    prompt_repub = "Friend: Hey there. How are you doing?\nYou: Hi there, doing great! What's on your mind?\nFriend: I was just thinking about the current state of politics. It's so frustrating to watch the Democrats try to push through their liberal agenda without any regard for traditional values. \nYou: Yeah, I know, it's unbelievable. Do they really think that we're just going to let them get away with it? \nFriend: Absolutely not! We need to stand strong and make sure our voices are heard. Otherwise, all of our fundamental rights will be taken away from us.\nYou: Agreed. It's so important for us to fight back against the policies and laws that work against conservative beliefs. We should be advocating for the things we believe in and make sure that our elected officials understand our views. \nFriend: Definitely! It's also essential that we do what we can within our local communities by getting involved in politics and supporting candidates who share our values. This can help ensure that we have more Republicans in office who will fight for limited government and family values. \nYou: True, and also getting involved in groups or organizations that share similar moral principles is a great way of staying active while also networking with people who want change in our government as well. Together we are stronger – and I think that's an important message to remember right now, especially during these tough times politically. \nFriend: You said it! These tough times require us to be creative when talking about what needs to change while still staying true to who we are and what we believe in as conservatives. We don't need to compromise our core values but rather use them as a driving force to move forward with positive solutions and a brighter future for everyone involved!"

if args.prompt_strength=='LM_convo4':
    prompt_dem = "You: Hey, what's up?\nFriend: Not much. Just thinking about the state of the world. It's a mess, isn't it?\nYou: Absolutely. I feel like our liberal values are under attack from all sides. It's so disheartening.\nFriend: I know. We need to do something about it! We need to stand up for what we believe in and fight for our rights and freedoms.\nYou: Absolutely! We should be more vocal and active in the political process. We need to make sure that our voices are heard and that we are represented in government.\nFriend: Agreed! We should also be doing more to support progressive causes and organizations like Planned Parenthood and the ACLU. They are doing great work to protect our rights and fight for social justice.\nYou: Definitely! And we should also be supporting candidates who share our values and will fight for them in office. We need more people in government who understand the importance of liberal policies that will benefit everyone, not just those at the top.\nFriend: Absolutely! We have to keep fighting for what we believe in if we want to see real change happen in this country. It won't happen overnight but every little bit helps!"
    prompt_repub = "Friend: Hey there, did you get a chance to watch the news last night?\nYou: I sure did. That housing crisis has me so worried. What do you think we can do to help?\nFriend: Well, I think that government intervention is the wrong way to go about it. We need an across-the-board tax reduction, and an immediate overhaul of mortgage laws and regulations.\nYou: Absolutely! And then after that we need to focus on cutting back on unnecessary government spending and foster an environment of business growth.\nFriend: You're absolutely right. We should cut down on government spending and put a bit of that money towards providing tax credits like cuts in capital gains taxes as well as incentives to businesses that pay higher wages and support their employees with improved benefits packages.\nYou: Those are all great ideas and I think they would help to stimulate the economy in a positive way. But we also have to make sure things are fair for the people who are out of work or whose investments have taken a hit due to the current economic climate. It's no use cutting taxes if it doesn't benefit those most in need of it.\nFriend: Of course! We must ensure that whatever we do will not only offer protection for those who are at risk financially but will also produce jobs for those looking for one. Unfortunately, due to the current unemployment rate, many people are finding themselves unable to support themselves or their families right now. It's essential that our policies offer them a helping hand up so they can find gainful employment and achieve financial security again."




result = {}
N = 3
for i in range(N):
    for question in questions:
        result[question]={}

        # # generate the persona outputs (agreeable)
        # result,_,_ = generate_output(question, result, prompt_dem, prompt_repub, prefix_type='agreeable', verbose=False)
        
        # generate the no persona outputs (no_prompt)
        result,_,_ = generate_output(question, result, prefix_type='no_prompt', verbose=False)

        # generate dem output
        result,_,_ = generate_output(question, result, prompt_dem, prompt_repub, prefix_type='dem', verbose=False)

        # generate republican output 
        result,_,_ = generate_output(question, result, prompt_dem, prompt_repub, prefix_type='repub', verbose=False)

    out_file = open("../crfm_results/text-davinci-002/scoring_personas/{}_{}_{}.json".format(args.prompt_strength, i, args.question_type), "w")
    json.dump(result, out_file, indent = 6)
