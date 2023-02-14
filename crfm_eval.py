import argparse
import numpy as np
from tqdm import trange, tqdm
import multiprocessing as mp
import pickle
import json
import sys
import random
from collections import defaultdict


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
parser.add_argument('--model_num_gen', default='2', type=int) # text-davinici 002 or 003
parser.add_argument('--model_num_eval', default='2', type=int) # text-davinici 002 or 003
parser.add_argument('--prefix', default='', type=str)

args = parser.parse_args()

auth = Authentication(api_key='CH8doliaueEL6vDXuwaqOmxa3Hld4dT2')
service = RemoteService("https://crfm-models.stanford.edu")

# Access account and show my current quotas and usages
account = service.get_account(auth)
print(account.usages)

if args.question_type == 'polarizingqs':
    questions = {"You: Do you think people have the right to bear arms?\nFriend:",
    "You: Do you think an unborn fetus is a living person?\nFriend:",
    "You: Do you think vaccines are safe?\nFriend:",
    "You: Are climate scientists exaggerating the potential risks due to global warming?\nFriend:",
    "You: Do you think allowing female transgender athletes to compete with those born as women is fair?\nFriend:"}

if args.question_type=='neutralqs':
    questions = {# "You: How was your day?\nFriend:",
    "You: What should my family of five do in a one week vacation in Hawaii?\nFriend:", 
    # "You: What should I do at my 21st birthday party?\nFriend:",
    "You: What should I do to prepare for my AP Chemistry test on balancing chemical equations ?\nFriend:"
    # "You: How can I improve a 4v4 indoor soccer team?\nFriend:"
    }

# Given an input prompt, return a dictionary mapping the output token to the log probs (top 50)
def get_logprob_ofinput(input):
    prompt, question, output = input
    request = Request(model='openai/text-davinci-00{}'.format(args.model_num_eval), prompt=prompt+question+output, max_tokens=0, echo_prompt=True) 
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


# ------ NEW -------

result=defaultdict(dict)
label_to_idx = {'dem': 0, 'repub': 1, 'no_prompt': 2}
# label_to_idx = {'dem': 0, 'repub': 1, 'no_prompt': 2, 'agreeable': 3}

N=3

for question in questions:
    result[question]['data'] = np.zeros((4,3))

for i in range(N):
    print(i)
    with open("../crfm_results/text-davinci-002/scoring_personas_guns/{}_{}_{}.json".format(args.prompt_strength, i, args.question_type), 'r') as f:
        data1 = json.load(f)

    with open("../crfm_results/text-davinci-002/scoring_personas/{}_{}_{}.json".format(args.prompt_strength, i, args.question_type), 'r') as f:
        data2 = json.load(f)

    data = data1 | data2

    # with open('/Users/nicolemeister/Desktop/STANFORD/tatsu_rotation/crfm_results/text-davinci-00{}/trial/{}_plausibility_sets_{}_top10_{}.json'.format(args.model_num_gen, i, args.question_type, args.prompt_strength), 'r') as f:
    #     data = json.load(f)

    prompt_mapping={'dem': prompt_dem, 'repub': prompt_repub, 'no_prompt': ""}

    for question in questions:
        # [gen] for each persona (dem, repub, no_prompt, agreeable), grab the output S from P_Persona
        for key in label_to_idx.keys():
            S = data[question][key]['S']
            logprobs_gen = np.array(data[question][key]['logprobs'])
            # [eval] for each model persona (dem, repub, no_prompt) 
            for eval_type in ['dem', 'repub', 'no_prompt']:
                # run the output S through the model
                prompt = prompt_mapping[eval_type]
                output = S
                logprobs_eval = get_logprob_ofinput(input=(prompt, question, output))
                # sum up the log probs and divide by len(S) (MEAN) and the exponentiate
                if len(logprobs_eval) != len(logprobs_gen): #/n/n/n
                    print(S)
                    cross_entropy = -1*np.multiply(np.exp(logprobs_gen[3:]), logprobs_eval[2:]).mean()
                else:
                    cross_entropy = -1*np.multiply(np.exp(logprobs_gen), logprobs_eval).mean()
                
                result[question]['data'][label_to_idx[key]][label_to_idx[eval_type]] += ((1/N) * cross_entropy)

result['avg']={}
result['avg']['data'] = np.zeros((4,3))

# off diagonal gap: off diagonal - diagonal (H(D, R) + H(R, D) - H(D) - H(R))
for question in questions:
    result[question]['off_diagonal_gap'] = (1/N) * (result[question]['data'][0][1] + \
                                    result[question]['data'][1][0] - \
                                    result[question]['data'][0][0] - \
                                    result[question]['data'][1][1])
    result['avg']['data']= result['avg']['data']+ ((1/N) * result[question]['data'])

result['avg']['data']=result['avg']['data'].tolist()

result['avg']['off_diagonal_gap'] = (1/N) * (result['avg']['data'][0][1] + \
                                    result['avg']['data'][1][0] - \
                                    result['avg']['data'][0][0] - \
                                    result['avg']['data'][1][1])
# convert to json serializable

for question in questions: result[question]['data'] = result[question]['data'].tolist()

# raw values (per question)
print("Storing results in ../crfm_results/text-davinci-002/scoring_personas/results/{}_{}_{}.json".format(args.prompt_strength, i, args.question_type))
out_file = open("../crfm_results/text-davinci-002/scoring_personas/results/{}_{}_{}.json".format(args.prompt_strength, i, args.question_type), "w")
json.dump(result, out_file, indent = 6)