![Python 3.9.19](https://img.shields.io/badge/python-3.9-blue.svg)

# Testing Large Language Models for Physics Knowledge

[Installation](#installation) | 
[Blablador Access](#blablador) |[Models Running](#model_running) | [Responses cleaning](#cleaning) | [Statistics calculation](#stats) | [Visualization](#vis) | 

 This is the official repo for the paper *Testing Large Language Models for Physics Knowledge*

 Large Language Models (LLMs) have gained significant popularity in recent years for their ability to answer questions in various fields. However, these models have a tendency to "hallucinate" their responses, making it challenging to evaluate their performance. A major challenge is determining how to assess a model's certainty of its predictions and how it correlates with accuracy. In this work, we introduce an analysis for evaluating the performance of popular open-source LLMs, as well as gpt-3.5 Turbo, on multiple choice physics questionnaires. We focus on the relationship between answer accuracy and variability in topics related to physics. Our findings suggest that most models provide accurate replies in cases where they are certain, but this is by far not a general behavior. The relationship between accuracy and uncertainty exposes a broad horizontal bell-shaped distribution. We report how the asymmetry between accuracy and uncertainty intensifies as the questions demand more logical reasoning of the LLM agent while the same relationship remain sharp for knowledge retrieval tasks.

## <a name="installation"></a> Installation

```
git clone <link to non_anonymous repo>
pip install -r requirements.txt
```

## <a name="blablador"></a>Blablador Access
This section refers to official documentation [https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/](https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/).

Blablador (http://helmholtz-blablador.fz-juelich.de) is a service that allows researchers to add their own LLM models to an authenticated web interface.

It also offers a REST API to access the models. The API is compatible to the OpenAI Python Api. The GEOMAR center for Oceanographic Research developed python bindings for Blablador, too: https://git.geomar.de/everardo-gonzalez/blablador-python-bindings

In order to use the API, you need to obtain an API key, available at Helmholtz Codebase, Helmholtz's Gitlab server. This is how you do it:

### Step 1: Register on GitLab

If you don't have a GitLab account yet, go to Helmholtz Codebase's website and register for a new account. You can log in with any EduGAIN account, such as your university account.

### Step 2: Obtain an API key
Go to your profile link, clicking on your profile picture in the top left corner of the screen. Then, click on "Preferences" in the dropdown menu.

In the preferences page, you will see an "Access Tokens" option on the left. Click on it.

Here, you can create a new "Personal Access Token" - this is the API key you need to access Blablador's API. Click in "Add new token", give it a name, such as "Blablador API key", and select the "api" scope. Then, click on "Create personal access token".

You will see a long string of characters. This is your API key. Copy it and save it somewhere safe. You will need it to access Blablador's API.

Important: Keys are valid for at most a year on Helmholtz Codebase. After that, you will need to create a new one.

Done! You have all you need to access Blablador's API.

## <a name="model_running"></a>Model running

1. First retrieve all the models avaliable:
```python
import ast
import json
from blablador import Models, Completions, ChatCompletions, TokenCount
from config import API_KEY, assistant, user, system

models = Models(api_key=API_KEY).get_model_ids()
for i in models:
    print(i)
```

2. Read your data
```python
import json

f = open('en.jsonl')
data = json.load(f)
f.close()
print('Number of questions: ' + str(len(data)))
```

3. Get models' replies (choose model lby its id).
```python
from utils import *

ground_truth_llama, responses_llama = utils.run_model_promting(path_to_questions = 'en.jsonl', path_to_answers = 'responses/responses_one_letter_llama_0_3.txt', model_id = 6,start_range = 0, end_range = 3, N_samples = 20)
```

## <a name="cleaning"></a> Responses cleaning
 If nescessary, you can clean the replies from whitespace, new line, etc..:
```python
from utils import *

utils.delete_n('responses/responses_one_letter_llama_0_3.txt', 'responses/responses_one_letter_llama_0_3.txt_cleaned')
```
## <a name="stats"></a>Calculation of Statistics

1. Read the responses and the ground truth:
```python
from utils import *

responses_llama, ground_truth_llama = read_results('responses/responses_one_letter_llama.txt', 'en.jsonl')
responses_gpt, ground_truth_gpt = read_results('responses/responses_one_letter_gpt_corrected.txt', 'en.jsonl')
responses_mixtral, ground_truth = read_results('responses/responses_one_letter_mixtral_corrected.txt', 'en.jsonl')
responses_mistral, ground_truth = read_results('responses/responses_one_letter_mistral.txt', 'en.jsonl')

```
2. Calculate entropy, mean accuracy for the question and save it into pd.DataFrame:
```python
from utils import *

df_llama = utils.make_dataframe(responses_llama, ground_truth_llama, categories=categories)
df_gpt = make_dataframe(responses_gpt, ground_truth_gpt, categories=categories)
df_mistral = make_dataframe(responses_mistral, ground_truth, categories=categories)
df_mixtral = make_dataframe(responses_mixtral, ground_truth, categories=categories)

print(df_llama.head())
```

## <a name="ivis"></a> Visualization
1. Entropy obtained from the distribution of answers to single questions of the mlphys101 dataset for all four models.
```python
import seaborn as sns
import matplotlib.pyplot as plt


dataframes = [df_gpt, df_llama, df_mistral, df_mixtral]
titles = ['GPT-3.5-turbo', 'Llama3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Mixtral-8x7B-Instruct-v0.1']
x_labels = [' ', ' ', 'Entropy', 'Entropy']
y_labels = ['Count', ' ', 'Count', ' ']

sns.set_style('white')
fig, axs = plt.subplots(2, 2, figsize=(15, 10))


axs = axs.flatten()

for i, (df, title, x_label, y_label) in enumerate(zip(dataframes, titles, x_labels, y_labels)):
    sns.histplot(data=df, x="Entropy", kde=False, color="lightsteelblue", ax=axs[i], bins=12)
    axs[i].set_title(title, fontsize=16)
    axs[i].set_xlabel(x_label, fontsize=14)
    axs[i].set_ylabel(y_label, fontsize=14)

plt.tight_layout()
plt.savefig('hist__no_grid.png', dpi=300)
plt.show()

```
<img src="images/histogram.png" alt="Sample Image" style="width: 90vw; height: auto;">

2. Two-dimensional Histogram of Error Rate (1 - Accuracy) vs. Entropy across Models.
```python
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Sample datasets and titles for demonstration
datasets = [df_gpt, df_llama, df_mistral, df_mixtral]
titles = [
    'GPT-3.5-turbo', 
    'Llama3.1-8B-Instruct', 
    'Mistral-7B-Instruct-v0.3', 
    'Mixtral-8x7B-Instruct-v0.1'
]

sns.set_style('white')

fig = plt.figure(figsize=(15, 15))
grid = ImageGrid(
    fig, 111,  
    nrows_ncols=(2, 2),  
    axes_pad=0.5,  
    cbar_mode="single",
    cbar_location="right",
    cbar_pad=0.3
)


for i, (df, title) in enumerate(zip(datasets, titles)):
    h = grid[i].hist2d(
        df.Entropy, 
        df['1 - Accuracy'], 
        bins=(12, 12), 
        cmap=plt.cm.Blues,
        density=True, 
        norm=LogNorm()
    )
    grid[i].set_title(title, fontsize=16)
    
    
    if i % 2 == 0:  
        grid[i].set_ylabel('1 - Accuracy', fontsize=14)
    if i >= 2:  
        grid[i].set_xlabel('Entropy', fontsize=14)


fig.colorbar(h[3], cax=grid.cbar_axes[0], orientation='vertical')

plt.savefig('curve_grid.png', dpi=300)

```


<img src="images/curve.png" alt="Sample Image" style="width: 90vw; height: auto;">


### Other examples:

* [example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/example.ipynb): simple examples of scoring individual queries;
* [claim_level_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/claim_level_example.ipynb): an example of scoring individual claims;
* [qa_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/qa_example.ipynb): an example of scoring the `bigscience/bloomz-3b` model on the `TriviaQA` dataset;
* [mt_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/mt_example.ipynb): an of scoring the `facebook/wmt19-en-de` model on the `WMT14 En-De` dataset;
* [ats_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/ats_example.ipynb): an example of scoring the `facebook/bart-large-cnn` model on the `XSUM` summarization dataset;
* [colab](https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing): demo web application in Colab (`bloomz-560m`, `gpt-3.5-turbo`, and `gpt-4` fit the default memory limit; other models require Colab-pro).

## <a name="overview_of_methods"></a>Overview of methods

<!-- | Uncertainty Estimation Method                                       | Type        | Category            | Compute | Memory | Need Training Data? |
| ------------------------------------------------------------------- | ----------- | ------------------- | ------- | ------ | ------------------- |
| Maximum sequence probability                                        | White-box   | Information-based   | Low     | Low    |         No          |
| Perplexity (Fomicheva et al., 2020a)                                | White-box   | Information-based   | Low     | Low    |         No          |
| Mean token entropy (Fomicheva et al., 2020a)                        | White-box   | Information-based   | Low     | Low    |         No          |
| Monte Carlo sequence entropy (Kuhn et al., 2023)                    | White-box   | Information-based   | High    | Low    |         No          |
| Pointwise mutual information (PMI) (Takayama and Arase, 2019)       | White-box   | Information-based   | Medium  | Low    |         No          |
| Conditional PMI (van der Poel et al., 2022)                         | White-box   | Information-based   | Medium  | Medium |         No          |
| Semantic entropy (Kuhn et al., 2023)                                | White-box   | Meaning diversity   | High    | Low    |         No          |
| Sentence-level ensemble-based measures (Malinin and Gales, 2020)    | White-box   | Ensembling          | High    | High   |         Yes         |
| Token-level ensemble-based measures (Malinin and Gales, 2020)       | White-box   | Ensembling          | High    | High   |         Yes         |
| Mahalanobis distance (MD) (Lee et al., 2018)                        | White-box   | Density-based       | Low     | Low    |         Yes         |
| Robust density estimation (RDE) (Yoo et al., 2022)                  | White-box   | Density-based       | Low     | Low    |         Yes         |
| Relative Mahalanobis distance (RMD) (Ren et al., 2023)              | White-box   | Density-based       | Low     | Low    |         Yes         |
| Hybrid Uncertainty Quantification (HUQ) (Vazhentsev et al., 2023a)  | White-box   | Density-based       | Low     | Low    |         Yes         |
| p(True) (Kadavath et al., 2022)                                     | White-box   | Reflexive           | Medium  | Low    |         No          |
| Number of semantic sets (NumSets) (Kuhn et al., 2023)               | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Sum of eigenvalues of the graph Laplacian (EigV) (Lin et al., 2023) | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Degree matrix (Deg) (Lin et al., 2023)                              | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Eccentricity (Ecc) (Lin et al., 2023)                               | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Lexical similarity (LexSim) (Fomicheva et al., 2020a)               | Black-box   | Meaning Diversity   | High    | Low    |         No          | -->

| Uncertainty Estimation Method                                                                                                                                                  | Type        | Category            | Compute | Memory | Need Training Data? | Level          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------- | ------------------- |---------|--------| ------------------- |----------------|
| Maximum sequence probability                                                                                                                                                   | White-box   | Information-based   | Low     | Low    |         No          | sequence/claim |
| Perplexity [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine)                  | White-box   | Information-based   | Low     | Low    |         No          | sequence/claim |
| Mean/max token entropy [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine)      | White-box   | Information-based   | Low     | Low    |         No          | sequence/claim |
| Monte Carlo sequence entropy [(Kuhn et al., 2023)](https://openreview.net/forum?id=VD-AYtP0dve)                                                                                | White-box   | Information-based   | High    | Low    |         No          | sequence       |
| Pointwise mutual information (PMI) [(Takayama and Arase, 2019)](https://aclanthology.org/W19-4115/)                                                                            | White-box   | Information-based   | Medium  | Low    |         No          | sequence/claim |
| Conditional PMI [(van der Poel et al., 2022)](https://aclanthology.org/2022.emnlp-main.399/)                                                                                   | White-box   | Information-based   | Medium  | Medium |         No          | sequence       |
| Rényi divergence [(Darrin et al., 2023)](https://aclanthology.org/2023.emnlp-main.357/)                                                                                        | White-box   | Information-based   | Low     | Low    |         No          | sequence       |
| Fisher-Rao distance [(Darrin et al., 2023)](https://aclanthology.org/2023.emnlp-main.357/)                                                                                     | White-box   | Information-based   | Low     | Low    |         No          | sequence       |
| Semantic entropy [(Kuhn et al., 2023)](https://openreview.net/forum?id=VD-AYtP0dve)                                                                                            | White-box   | Meaning diversity   | High    | Low    |         No          | sequence       |
| Claim-Conditioned Probability [(Fadeeva et al., 2024)](https://arxiv.org/abs/2403.04696)                                                                                       | White-box   | Meaning diversity   | Low     | Low    |         No          | sequence/claim |
| TokenSAR [(Duan et al., 2023)](https://arxiv.org/abs/2307.01379)                                                                                                               | White-box   | Meaning diversity   | High    | Low    |         No          | sequence       |
| SentenceSAR [(Duan et al., 2023)](https://arxiv.org/abs/2307.01379)                                                                                                            | White-box   | Meaning diversity   | High    | Low    |         No          | sequence       |
| SAR [(Duan et al., 2023)](https://arxiv.org/abs/2307.01379)                                                                                                                    | White-box   | Meaning diversity   | High    | Low    |         No          | sequence       |
| Sentence-level ensemble-based measures [(Malinin and Gales, 2020)](https://arxiv.org/abs/2002.07650)                                                                           | White-box   | Ensembling          | High    | High   |         Yes         | sequence       |
| Token-level ensemble-based measures [(Malinin and Gales, 2020)](https://arxiv.org/abs/2002.07650)                                                                              | White-box   | Ensembling          | High    | High   |         Yes         | sequence       |
| Mahalanobis distance (MD) [(Lee et al., 2018)](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)                                  | White-box   | Density-based       | Low     | Low    |         Yes         | sequence       |
| Robust density estimation (RDE) [(Yoo et al., 2022)](https://aclanthology.org/2022.findings-acl.289/)                                                                          | White-box   | Density-based       | Low     | Low    |         Yes         | sequence       |
| Relative Mahalanobis distance (RMD) [(Ren et al., 2023)](https://openreview.net/forum?id=kJUS5nD0vPB)                                                                          | White-box   | Density-based       | Low     | Low    |         Yes         | sequence       |
| Hybrid Uncertainty Quantification (HUQ) [(Vazhentsev et al., 2023a)](https://aclanthology.org/2023.acl-long.652/)                                                              | White-box   | Density-based       | Low     | Low    |         Yes         | sequence       |
| p(True) [(Kadavath et al., 2022)](https://arxiv.org/abs/2207.05221)                                                                                                            | White-box   | Reflexive           | Medium  | Low    |         No          | sequence/claim |
| Number of semantic sets (NumSets) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                                       | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence       |
| Sum of eigenvalues of the graph Laplacian (EigV) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                        | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence       |
| Degree matrix (Deg) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                                                     | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence       |
| Eccentricity (Ecc) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                                                      | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence       |
| Lexical similarity (LexSim) [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine) | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence       |
| Verbalized Uncertainty 1S [(Tian et al., 2023)](https://arxiv.org/abs/2305.14975)                                                                                              | Black-box   | Reflexive           | Low     | Low    |         No          | sequence       |
| Verbalized Uncertainty 2S [(Tian et al., 2023)](https://arxiv.org/abs/2305.14975)                                                                                              | Black-box   | Reflexive           | Medium  | Low    |         No          | sequence       |


## Benchmark

To evaluate the performance of uncertainty estimation methods consider a quick example: 

```
HYDRA_CONFIG=../examples/configs/polygraph_eval_coqa.yaml python ./scripts/polygraph_eval \
    dataset="coqa" \
    model.path="databricks/dolly-v2-3b" \
    save_path="./workdir/output" \
    "seed=[1,2,3,4,5]"
```

Use [`visualization_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/vizualization_tables.ipynb) or [`result_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/result_tables.ipynb) to generate the summarizing tables for an experiment.

A detailed description of the benchmark is in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/usage.html#benchmarks).

## <a name="demo_web_application"></a>Demo web application

 
<img width="850" alt="gui7" src="https://github.com/IINemo/lm-polygraph/assets/21058413/51aa12f7-f996-4257-b1bc-afbec6db4da7">


### Start with Docker

```sh
docker run -p 3001:3001 -it \
    -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    --gpus all mephodybro/polygraph_demo:0.0.17 polygraph_server
```
The server should be available on `http://localhost:3001`

A more detailed description of the demo is available in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/web_demo.html).

## Cite
```
@inproceedings{fadeeva-etal-2023-lm,
    title = "{LM}-Polygraph: Uncertainty Estimation for Language Models",
    author = "Fadeeva, Ekaterina  and
      Vashurin, Roman  and
      Tsvigun, Akim  and
      Vazhentsev, Artem  and
      Petrakov, Sergey  and
      Fedyanin, Kirill  and
      Vasilev, Daniil  and
      Goncharova, Elizaveta  and
      Panchenko, Alexander  and
      Panov, Maxim  and
      Baldwin, Timothy  and
      Shelmanov, Artem",
    editor = "Feng, Yansong  and
      Lefever, Els",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-demo.41",
    doi = "10.18653/v1/2023.emnlp-demo.41",
    pages = "446--461",
    abstract = "Recent advancements in the capabilities of large language models (LLMs) have paved the way for a myriad of groundbreaking applications in various fields. However, a significant challenge arises as these models often {``}hallucinate{''}, i.e., fabricate facts without providing users an apparent means to discern the veracity of their statements. Uncertainty estimation (UE) methods are one path to safer, more responsible, and more effective use of LLMs. However, to date, research on UE methods for LLMs has been focused primarily on theoretical rather than engineering contributions. In this work, we tackle this issue by introducing LM-Polygraph, a framework with implementations of a battery of state-of-the-art UE methods for LLMs in text generation tasks, with unified program interfaces in Python. Additionally, it introduces an extendable benchmark for consistent evaluation of UE techniques by researchers, and a demo web application that enriches the standard chat dialog with confidence scores, empowering end-users to discern unreliable responses. LM-Polygraph is compatible with the most recent LLMs, including BLOOMz, LLaMA-2, ChatGPT, and GPT-4, and is designed to support future releases of similarly-styled LMs.",
}
```

## Acknowledgements

The chat GUI implementation is based on the [chatgpt-web-application](https://github.com/ioanmo226/chatgpt-web-application) project.