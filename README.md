# LLMs for the two long integers addition 

Clone thos repository and create an environment using requirements.txt:
```
conda create --name <env> --file requirements.txt
```
or
```
pip install -r requirements.txt
```

# Problem description

Adapt a large language model (LLM) (not more than 4B parameters) to solve the problem of addition of two long integers (as many digits as possible).
The quality will be measured on a randomly generated set of numbers of different lengths.



# Solutions

## 1. First approach - teach a model with demonstrations

### Literature review
The first possible direction of solving the problem of addition of two long integers as one of the mathematical problems is to teach language models the process of reasoning.
Recently, LLMs have shown impressive success on a wide range of tasks by using this approach.

For example, paper [[1]](#1) demonstrates that by **fine-tuning** an autoregressive language model (GPT-Neo) on appropriately structured step-by-step demonstrations, it is possible to teach it to execute 
a mathematical task that has previously proved difficult for Transformers – longhand modulo  operations – with a relatively small number of examples. 
 Their demonstrations are presented in a widely used **chain-of-thought (COT) form**.


Nevertheless, paper [[6]](#6) shows that while LLMs can 'themselves' perform simple arithmetic operations, their performance falls dramatically when dealing with large numbers. 
As shown in [[7]](#7),  even when fine-tuning a PaLM-based model (Pathways Language Model [[5]](#5)) on 164B tokens of explicit mathematical content, one of its two
most common failures is “incorrect calculation”. That's why it is worth considering a slightly different approach.

Papers [[2]](#2) and [[3]](#3) go in approximately in the same direction as [[1]](#1) but use **few-shot** prompting ([[4]](#4)) in some kind of **algorithmic form**.

- Proposed in [[3]](#3) approach uses the LLM to read natural language problems and generate programs as the intermediate
reasoning steps, but offloads the solution step to a runtime (**Python interpreter**)

- Proposed in [[2]](#2) approach combines an LLM that can incrementally formalize word problems as a set of variables and 
equations with an external **symbolic solver** that can solve them


### Solution description: few-shot promting & using a python interpreter

#### Motivation
My first solution was mostly inspired by and based on paper [[3]](#3). Here are some reasons:

1. It has been shown in [[2]](#2) that their approach is more effective for more difficult problems that require declarative reasoning while the 
results on simple tasks like ours (addition of two integers) [[2]](#2) and [[3]](#3) show comparable results.


2. Since python 3 has [no more limit to value of integers](https://docs.python.org/3/whatsnew/3.0.html#integers), almost all limits for the length of
 numbers to sum are removed. The only limitation is the maximum length of the model's output.
   

3. Large Numbers or Incorrect Reasoning? In [[3]](#3) authors show that 
   the primary failure mode during working with large numbers is the inability to perform such arithmetic accurately, not the wrong generated solution steps  
   So the main thing to focus on is performing arithmetic accurately which is esay to do using python.

   
4. In [[3]](#3) authors show that their approach PAL (Program-Aided Language models) can work with weaker models (while its benefit over chain-of-thought scales elegantly to stronger models as well). As far as we have a limitation of model size (not more than 4B parameters) in the task, this is an important inference.



#### Few-shot idea

Few-shot prompting does not require task-specific fine-tuning of the base model, so it does not modify the underlying LLM.
It leverages the strength of large-language
models to solve a task with a set of k examples that are provided as part of the test-time input, where k is usually a number in the low single digits ([[4]](#4)). 
These input-output  examples {(xi, yi)}, i=1;k are concatenated in a prompt p ≡ (x1 · y1) || (x2 · y2) || ... || (xk · yk). where “·” denotes
the concatenation of an input and output, and “||” indicate
the concatenation of different examples. During inference,
a test instance xtest is appended to the prompt, and p || xtest
is passed to the model which attempts to complete p || xtest,
and thereby generate an answer ytest.

#### Prompting
My prompt consists of 4 examples of the task of addition of two numbers (both positive, first negative and second positive, first positive and second negative, both negative) with their code solutions and the target task question.
For example, if we need to add 2 and 3 then the prompt would be:

```
Q: What is 121231 plus 349340?
    
# solution in Python:
def solution():
    return 121231 + 349340
    
    
Q: What is -12 plus 31323?

# solution in Python:
def solution():
    return -12 + 31323
    
    
Q: What is 93 plus -2901201?

# solution in Python:
def solution():
    return 93 + -2901201


Q: What is -123 plus -132?
# solution in Python:
def solution():
    return -123 + -132


Q: What is 2 plus 3?
# solution in Python:
```

(using fewer examples it is possible to reduce the response time of the model. However, solution evaluation shows tha at least these 4 examples (with all combinations of positive and negative summands) should be shown. Otherwise, the model fails mostly on the tasks that have a combination that it has not seen)

#### Base model
In the original paper authors mostly used CODEX (specifically, code-davinci-002) as a backend LLM. They were descendants of GPT-3 models that would understand and generate code. 
However, Codex models are now [deprecated](https://platform.openai.com/docs/models/codex). Moreover, all of them do not satisfy the limitation on the number of base LLM parameters (4B).

In original paper [[3]](#3) authors show that this approach performs better when the base model either has high 'code modeling ability' or
it is a 'sufficiently strong' natural language model (which requires billions of parameters). 

That's why it was decided to use a model with high “code modeling ability” and way less than 4B params:
[Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono)

Information about the model:
```
CodeGen is a family of autoregressive language models for program synthesis 
from the paper: A Conversational Paradigm for Program Synthesis by Erik Nijkamp,
Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong. 

The checkpoint included in this repository is denoted as CodeGen-Mono 350M 
in the paper, where "Mono" means the model is initialized with CodeGen-Multi 350M 
and further pre-trained on a Python programming language dataset, 
and "350M" refers to the number of trainable parameters.
```

#### Solution steps
Given two numbers for addition:

1. Create the described prompt 
   (see `solution1_few_shot_with_python/prompt_generation.py`)
2. Send the prompt to the model and get its response 
   (see Solver.calc_sum in `solution1_few_shot_with_python/solver.py`)
3. Select a section of the response where the generated code for the solution of the target task is
   (see Solver.calc_sum in `solution1_few_shot_with_python/solver.py`)
4. Execute the code with (my own) runtime
   (see Solver.execute in `solution1_few_shot_with_python/solver.py` and `solution1_few_shot_with_python/runtime.py`)
5. Return the result of the execution as answer
   (see Solver.calc_sum in `solution1_few_shot_with_python/solver.py`)


#### Usage

(the response may take a long time)

a) Run command in the command line from /solution1_few_shot_with_python 
``` 
python main.py --a=2 --b=3
```
where `a` and `b` are the two integers to sum.
The result would be a string: `a + b = x` where a, b - given numbers and x is the produced result.

b) Use Solver1 class from /solution1_few_shot_with_python/solver:
```
solver = Solver1()
a = 2
b = 3
answer_int, meta_info = solver.calc_sum(a, b)
```
Its main method calc_sum takes as arguments two integers to sum and returns the integer result of summation and a dict with meta_info (i.e. the created promt, full model response, chosen code)



#### Interesting problems found
Given a number that is a kind of cyclical (i.e. 123456789012345678901234567890) the solution may fail as the model often decides to continue the detected cycle instead of generating the code
 
#### Evaluation
The data for the evaluation was generated with script `solutions_evaluation/generate_numbers_to_sum.py`
With the maximum amount of digits in the number `cur_d` from 10 to 100 (with step 10): for each `cur_d` the following examples were randomly generated
* when both numbers have exactly `cur_d` digits 
* when one number has exactly `cur_d` digits and the other one -- from 1 to `cur_d` digits

In each described pattern each number was randomly made positive or negative and 5 examples of each pattern were created. This results in 100 examples in the file `solutions_evaluation/test_examples.jsonl`

The evaluation was produced by script `solutions_evaluation/evaluate.py`.

The chosen metric is accuracy (with the requirement that the result of the solution must *exactly* match the expected)

The results of the evaluation (with the solution answers, the expected result, time spent, generated code for each test example) can be found in in the file `solutions_evaluation/test_examples_solution1_results.jsonl`

The solution accuracy is 0.97.

In two examples out of three where the solution failed the problem was that the runtime could not evaluate the generated code (with cur_d 30 and 70), and in the last one -- the answer was wrong: different sign and absolute value (with cur_d=40) 



## 2. Second approach - improve the ability of LLMs to perform addition operation

### Literature review


With transformers, the only input to the model is the surface form of text combined with supplemental embeddings.
In [[6]](#6) authors found that how a number is represented in its surface form has a strong influence on the model’s accuracy.
They showed  that it is possible to “inject”  representations into transformer models by simple manipulations of the input sequence (in their case,
explicitly enumerating the semantics of digit positions), without any need to re-pretrain.



However, authors conclude that regardless of the number of parameters and training examples, models cannot seem to learn addition rules that are independent of the length of the
numbers seen during training.  Models cannot extrapolate, i.e., they fail to perform simple arithmetic when
evaluated on inputs whose length distribution differs from the one seen during training. This appears
to be a problem that neither larger models, more compute, nor more data can solve.




My solution is based on [[6]](#6), taking into account the results of their experiments with the underlying LLMs, the form of representation of numbers, training data, etc.

### Solution description: train T5 on arithmetic problems.

First, we will discuss the process of training the model

#### 0. Question form
Training, development, and test sets are programmatically generated. The input template is always “What is [number1] plus [number2]?”, where 
[number1] and [number2] are numbers randomly sampled. 

#### 1. How sets are sampled?
Authors experimented with two methodologies:
  
Balanced sampling: To generate sets, first set the maximum number
of digits D and then create each example as follows: first sample d from [2, D] and then independently sample [number1] and [number2] from
[10^(D−1) , (10^D) − 1]. This method ensures that the set will
have a roughly equal proportion of d-digit numbers, where d ∈ [2, D].

Random sampling: To generate sets, sample [number1] and [number2] independently from
[0, (10^D)−1]. This results in approximately 90% of the numbers having D-digits, 9% having (D−1)-
digits, and so on. This unbalanced set aims at evaluating models on the largest numbers it was trained
on. 

The results of experiments show that when trained on the balanced distribution, the model succeeds on
both random and balanced evaluation sets. When trained on the random distribution, it succeeds on
the random evaluation set, but it fails on the balanced evaluation set

That's why in my solution model is trained on the **balanced distribution**.


#### 2. How numbers are represented?
Authors experimented with 6 different number representations:

```
Orthography          Example                    Notes
DECIMAL              832                        default representation
CHARACTER            8 3 2                      ensures consistent tokenization
FIXED-CHARACTER      0 8 3 2                    ensures consistent positions (e.g., max. 4 digits)
UNDERSCORE           8_3_2                      underscores provide hints on digit significance
WORDS                eight hundred thirty-two   leverages pretraining
10-BASED             8 100 3 10 2               easy to determine digit significance
10E-BASED            8 10e2 3 10e1 2 10e0       more compact encoding of above
```

Experiments show that:
* With up to 15 digits, the 10-BASED and 10E-BASED representations achieve accuracy close to 100%.
Authors explanation for their success is the explicit position tokens added between each digit, which
allows the model to inspect the left or right tokens of a digit to determine its significance.


* Authors show that with 10E-BASED representation the model learns to accurately add and subtract numbers up to 60 digits (and that's the best result within all types of representation)


That's why in my solution numbers are represented in the **10E-BASED** form.

#### 3. Regular or inverse order?

Auto-regressive models  generate the output sequence token by token. Thus, to produce the first digit of the answer, which is the most
significant one, the model has to perform all the carry operations.  Hence, the model has to
perform the digit-wise addition (or subtraction) of all the digits in the question before generating the
first digit of the answer. Authors call this generation order “regular”.

Another way to produce an answer is by generating the least significant digits first. This order is
perhaps easier to learn than the “regular” order because to decode each digit, the model only needs
to add (or subtract) single digits and check if the previous digit-wise operation had a carry. Authors call
this generation order “inverse”.

*(For interpolation experiments, the models are trained and evaluated on up to 60-digit numbers. For
extrapolation experiments, the models are trained on up to 50-digit numbers and evaluated on 60-
digit numbers.)*

Experiments results show that regardless of the model size, performed operations, amount of digits in numbers,
the difference in accuracy is negligible between regular and inverse orders on interpolation tasks.
However, models trained and evaluated on the regular order show higher extrapolation accuracy
than those that use the inverse order.



That's why in my solution numbers are represented in the **regular order**.

*This result is perhaps surprising since one would expect that the inverse order would be easier to learn. Possible explanation: In the inverse order, the answer is generated from least to
most significant digit, so the model might have a tendency to select the termination token right after
it generates the most significant digit seen during training. In the regular order, however, the model
has to predict the full length of the sequence before emitting the first and second tokens.*


#### Model size

Experiments show that larger models might perform better on data whose distribution is
outside its training data distribution. Although larger models perform better than smaller ones, authors show that not
even 3B-parameter models can learn simple arithmetic rules for infinitely large numbers. Extrapolation is hardly achieved when trained on fewer than 50 digits, regardless of the model size

Due to the limitation of model size in the task and my limited computing resources, **I use [t5-base model](https://huggingface.co/t5-base) with 220 million parameters as a base model** with the expectation that with an increase in the number of parameters in the base model (i.e. using [t5-large](https://huggingface.co/t5-large) model with 770 million parameters) the performance will increase.



#### Data size
Authors show that beyond a critical amount, increasing the training data does not improve extrapolation accuracy. 

That's why I use a **training dataset with 100000 examples** as recommended by the authors


#### Training steps amount
As
training progresses, interpolation accuracy always reaches 100%, but extrapolation accuracy starts
to decrease after some number of training steps. The number of training steps after which this drop
occurs varies dramatically between runs that differ only in the seed used to generate the training data.
That's why checkpoints with the best performance are saved after each epoch end.

№№ here
#### Interesting observation
Contrary to the hypothesis of Newman et al. (2020), we find that the end-of-sequence token does
not seem to be the cause of extrapolation failures. For example, when a T5-770M model trained on
30-digit numbers is evaluated on 60-digit numbers, it correctly generates the first 23 position tokens
(i.e., from “10e60” until “10e38”) but it suddenly skips to position token “10e27”, and continues
generating the correct position tokens until the last one (“10e0”). Here we show one such sequence:
```
1 10e60 0 10e59 1 10e58 2 10e57 3 10e56 0 10e55 2 10e54 7 10e53 0 10e52
1 10e51 0 10e50 3 10e49 9 10e48 0 10e47 5 10e46 3 10e45 1 10e44 5 10e43 3
10e42 6 10e41 3 10e40 6 10e39 0 10e38 8 10e27 1 10e26 4 10e25 1 10e24 2 10e23
6 10e22 6 10e21 9 10e20 5 10e19 3 10e18 4 10e17 8 10e16 3 10e15 8 10e14 8
10e13 9 10e12 5 10e11 3 10e10 5 10e9 0 10e8 6 10e7 4 10e6 3 10e5 5 10e4 6
10e3 7 10e2 2 10e1 2 10e0
```
Hence, although the model correctly emits the end-of-sequence token after the “10e0” token, it
decides to shorten the sequence in the middle of the generation, i.e., by skipping position tokens
“10e37” until “10e28”. This skipping behavior is consistent across model sizes, dataset sizes, and
extrapolation ranges (e.g., training on 20 digits, evaluating on 30 digits, etc.). Investigating it further
might help us understand why neural models often fail on extrapolation tasks.

**описать остальные параметры**

** IMPACT OF DATA SIZE**
## INVESTIGATING_THE_LIMITATIONS_OF_TRANSFORM_apr21


**увеличение числа цифр**
One advantage of working with arithmetic tasks is that the rules to be learned are well defined and
relatively simple. Thus, it is easy to verify if models learned such rules by evaluating them on
numbers that are larger than the ones they were trained on. If successful, such a model would have
no problem correctly adding or subtracting arbitrarily long numbers


**обучать с как минимум 50 цифрами**
Extrapolation is hardly achieved when trained on fewer than 50 digits, regardless of the model size






## References
<a id="1">[1]</a> 
Gabriel Recchia. Teaching Autoregressive Language Models Complex Tasks By Demonstration. CoRR, abs/2109.02102, 2021. https://arxiv.org/abs/2109.02102

<a id="2">[2]</a> 
He-Yueya, J., Poesia, G., Wang, R. E., & Goodman, N. D. Solving Math Word Problems by Combining Language Models With Symbolic Solvers, 2023. https://arxiv.org/abs/2304.09102


<a id="3">[3]</a> 
Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., & Neubig, G. PAL: Program-aided Language Models, 2023.  https://arxiv.org/abs/2211.10435

<a id="4">[4]</a> 
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam,
G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural
information processing systems, 33:1877–1901, 2020. https://arxiv.org/abs/2005.14165

<a id="5">[5]</a> 
Chowdhery, A., Narang, S., et al. PaLM: Scaling Language Modeling with Pathways. arXiv preprint
arXiv:2204.02311, 2022.  https://arxiv.org/abs/2204.02311

<a id="6">[6]</a> 
Nogueira, R., Jiang, Z., & Lin, J. Investigating the Limitations of Transformers with Simple Arithmetic Tasks, 2021. https://arxiv.org/abs/2102.13019

<a id="7">[7]</a> 
Lewkowycz, A., Andreassen, A., et al. Solving quantitative reasoning problems with language models. arXiv preprint
arXiv:2206.14858, 2022. https://arxiv.org/abs/2206.14858
