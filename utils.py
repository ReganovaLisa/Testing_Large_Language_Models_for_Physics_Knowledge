import ast
import json
from blablador import Models, Completions, ChatCompletions, TokenCount
from config import API_KEY, assistant, user, system
from scipy.stats import entropy
import numpy as np
import plotly.express as px
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import top_k_accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme(style='white')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def one_letter_answer(text, model_id):
    models = Models(api_key=API_KEY).get_model_ids()
    completion = ChatCompletions(api_key=API_KEY, model=models[model_id])
    response = completion.get_completion(messages=[
        system("You’re a highly knowledgeable physics tutor. For each message, give only the letter of the correct answer without any explanations or additional information."),
        user("A ball rolls down a slope and accelerates uniformly at 2 m/s². If it starts from rest, what will be its speed after 3 seconds? A. 3 m/s, B. 4 m/s, C. 5 m/s, D. 6 m/s, E. 7 m/s"),
        assistant("D"),
        user("A cyclist accelerates uniformly from rest to a speed of 10 m/s in 5 seconds. What is their acceleration? A. 1 m/s², B. 2 m/s², C. 3 m/s², D. 4 m/s², E. 5 m/s²r"),
        assistant("B"),
        user("A rocket accelerates from rest at a constant rate of 6 m/s². What speed will it reach after 4 seconds? A. 12 m/s, B. 18 m/s, C. 24 m/s, D. 30 m/s, E. 36 m/s"),
        assistant("C"),
        user(text),
    ])
    return response

def calculate_probabilities(responses):
    probabilities = []
    for i in range(len(responses)):
        num_enc = [0,0,0,0,0]
        N_samples = len(responses[0])
        for j in range(N_samples):
            if responses[i][j] == "A":
                num_enc[0] += 1
            elif responses[i][j] == "B":
                num_enc[1] += 1
            elif responses[i][j] == "C":
                num_enc[2] += 1
            elif responses[i][j] == "D":
                num_enc[3] += 1
            elif responses[i][j] == "E":
                num_enc[4] += 1
        num_enc = np.array(num_enc)
        probabilities.append(num_enc/N_samples)
        
    return np.array(probabilities)

def responses_to_numbers(responses):
    resp_in_numbers = []
    for i in range(len(responses)):
        num_enc = []
        for j in range(len(responses[i])):
            if responses[i][j] == "A":
                num_enc.append(0)
            elif responses[i][j] == "B":
                num_enc.append(1)
            elif responses[i][j] == "C":
                num_enc.append(2)
            elif responses[i][j] == "D":
                num_enc.append(3)
            elif responses[i][j] == "E":
                num_enc.append(4)
            elif responses[i][j] == "None":
                num_enc.append('None')
        resp_in_numbers.append(num_enc)
    return resp_in_numbers

def run_model_promting(path_to_questions, path_to_answers,model_id,start_range, end_range, N_samples = 20):
    replies = []
    ground_truth = []
    f = open(path_to_questions)
    data = json.load(f)
    f.close()
    
    f = open(path_to_answers, 'w')
    print('Model name: ' + models[model_id])
    

    for i in range(start_range,end_range):
        one_q_replies = []
        question = data[i]['task']
        ground_truth.append(data[i]['ans']) 

        for j in range(len(data[i]['opt'])):
            question += data[i]['opt'][j]

        for k in range(N_samples):
            response = one_letter_answer(question, model_id = model_id)
            #print(response)
           
            one_q_replies.append(ast.literal_eval(response)["choices"][0]["message"]['content'])
        
        replies.append(one_q_replies)
        f.write(f"{one_q_replies}\n")
        

    f.close()

    return ground_truth, replies

def filter_single_letter(strings):

    pattern = re.compile(r'^\s*\n?[A-E]\n?\s*$')

    # Create a new list with filtered values or None for non-matching entries
    filtered_strings = [s.strip() if pattern.fullmatch(s.strip()) else None for s in strings]

    return filtered_strings
    
def delete_n_(path_to_answers, path_to_answers_one_letter):
    replies_one_letter = []
    
    with open(path_to_answers) as f_resp:
        responses = f_resp.read().splitlines()
    for i in range(len(responses)):
        responses[i] = ast.literal_eval(responses[i])
    
    f = open(path_to_answers_one_letter, 'w')
    
    

    for i in range(len(responses)):
       
        resp =filter_single_letter(responses[i])
 
       
        f.write(f"{resp}\n")
        

    f.close()

    return replies_one_letter

def read_results(path_to_answers, path_to_dataset):
    with open(path_to_answers) as f:
        responses = f.read().splitlines()
    for i in range(len(responses)):
        responses[i] = ast.literal_eval(responses[i])
    

    fi = open(path_to_dataset)
    data = json.load(fi)
    fi.close()
    ground_truth = []

    for i in range(len(responses)):
        question = data[i]['task']
        ground_truth.append(data[i]['ans']) 
    
    return responses, ground_truth

# this function calculates entropy for question responses
# not correct questions but how different replies for the same question 
def get_entropy(one_q_reply, possible_answers = ['A', 'B', 'C', 'D', 'E']):
    filtered_replies = [reply for reply in one_q_reply if reply is not None]
    
    if not filtered_replies:
        return 0  
    letter_counts = Counter(filtered_replies)
    total_count = len(filtered_replies)
    
    probability_distribution = [letter_counts.get(letter, 0) / total_count for letter in possible_answers]
    
    H = entropy(probability_distribution)
    return H

### let's name the accuracy as a number of correct answers/N_samples
def mean_accuracy(responses, ground_truth):
    res = []
    N_s = len(responses[0])
    
    for i in range(len(responses)):
        response_array = np.array(responses[i])
        valid_mask = (response_array != None) 
        corr_num = np.sum(response_array[valid_mask] == ground_truth[i])
        valid_count = np.sum(valid_mask)
        accuracy = corr_num / valid_count if valid_count > 0 else 0  
        res.append(accuracy)   
    return res
def make_dataframe(responses, ground_truth, categories):
    
    entropy_list = []
    for i in range(len(responses)):
        entropy_list.append(get_entropy(one_q_reply=responses[i]))

    responses_num = responses_to_numbers(responses)
    acc = mean_accuracy(responses_num, ground_truth)
    dict_df = {'Accuracy': np.array(acc),'1 - Accuracy': 1 - np.array(acc), 'Entropy': entropy_list, 'Category': categories} 
    df = pd.DataFrame(dict_df)
    return df

def retrieve_categories(questions):
    categories = []
    for i in range(len(questions)):
        cat = questions[i]['diff']
        categories.append(cat)
    return  categories

