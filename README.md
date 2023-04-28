# LLMs for the two long integers addition 


# Problem description

Adapt a large language model (not more than 4B parameters) to solve the problem of addition of two long integers (as any digits as possible).
The quality will be measured on a randomly generated set of numbers of different lengths.



# Solutions

## 1. First approach - teach a model with demonstrations

### Literature review
The first possible direction of solving the problem of addition of two long integers as one of the mathematical problems is to 'teach' language models the process of reasoning.
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
results on simple tasks like ours (addition of two long integers) [[2]](#2) and [[3]](#3) show comparable results.


2. Since python 3 has [no more limit to value of integers](https://docs.python.org/3/whatsnew/3.0.html#integers), almost all limits for the length of
 numbers to sum are removed. The only limitation is the maximum length of the model's output.
   

3. Large Numbers or Incorrect Reasoning? In [[3]](#3) authors show that 
   the primary failure mode during working with large numbers is the inability to perform such arithmetic accurately, not the wrong generated solution steps  
   So the main thing to focus on is performing arithmetic accurately which is esay to do using python.

   
4. In [[3]](#3) authors show that their approach PAL (Program-Aided Language models) can work with weaker models, while its benefit over chain-of-thought scales elegantly to stronger models as well. As far as we have a limitation of model size (not more than 4B parameters) in the task, this is an important inference.



#### Few-shot idea

Few-shot prompting does not require task-specific fine-tuning of the base model, so it does not modify the underlying LLM
It leverages the strength of large-language
models to solve a task with a set of k examples that are provided as part of the test-time input, where k is usually a number in the low single digits ([[4]](#4)). 
These input-output  examples {(xi, yi)}, i=1;k are concatenated in a prompt p ≡ (x1 · y1) || (x2 · y2) || ... || (xk · yk). where “·” denotes
the concatenation of an input and output, and “||” indicate
the concatenation of different examples. During inference,
a test instance xtest is appended to the prompt, and p || xtest
is passed to the model which attempts to complete p || xtest,
and thereby generate an answer ytest. Note that such few-shot prompting .

#### Prompting
My prompt consists of 4 examples of the task of addition of two numbers (both positive, first negative and second positive, first positive and second negative, both negative) with their code solutions and the *target* task question.
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

(using fewer examples it is possible to reduce the response time of the model. However solution evaluation shows tha at least theese 4 examples (with all combinations of positive and negative summands) should be shown. Otherwise, the model fails mostly on the tasks that have a combination that it has not seen)

#### Base model
In the original paper authors mostly used CODEX (specifically, code-davinci-002) as a backend LLM. However Codex models are now [deprecated](https://platform.openai.com/docs/models/codex). Moreover, all of them do not satisfy the limitation on the number of base LLM parameters (4B).
They were descendants of our GPT-3 models that would understand and generate code. 

In original paper [[3]](#3) authors show that this approach performs better when the base model either has high 'code modeling ability' or
it is a 'sufficiently strong' natural language model (which requires billions of parameters). 

That's why it was decided to use a model with high “code modeling ability” and way less than 4B params:
[Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono)


#### Solution steps
Given to numbers for addition:

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

a) Run from /solution1_few_shot_with_python in the command line command
``` 
python main.py --a=2 --b=3
```
where `a` and `b` are the two integers to sum

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

The solution accuracy is 


## INVESTIGATING_THE_LIMITATIONS_OF_TRANSFORM_apr21
https://github.com/castorini/transformers-arithmetic

* Как представлять: By introducing position tokens (e.g., “3 10e1 2”), the
model learns to accurately add and subtract numbers up to 60 digits.

* subword tokenizers and positional encodings are components in current transformer designs that might need improvement. Moreover, we
show that regardless of the number of parameters and training examples, models cannot seem to learn addition rules that are independent of the length of the
numbers seen during training.

**введение**

в трансформерах подается непонятно что и на что мы можем влиять -- With transformers, the only input to the model
is the surface form of text combined with supplemental embeddings

Это плохо

Our work shows that it is possible to “inject”
representations into transformer models by simple manipulations of the input sequence (in our case,
explicitly enumerating the semantics of digit positions)

Причем это без переобучения


Но кажется все  плохо -- если не обучались на длинных, то ничего не получится

 Despite our best
efforts, we find that models cannot extrapolate, i.e., they fail to perform simple arithmetic when
evaluated on inputs whose length distribution differs from the one seen during training. This appears
to be a problem that neither larger models, more compute, nor more data can solve.

О том, что все плохо как раз и пишут в статье PAL и поэтому ищут другие подходы

**в каком из 6 видов представлять числа**
10E-BASED
причем without target position encoding

**увеличение числа цифр**
One advantage of working with arithmetic tasks is that the rules to be learned are well defined and
relatively simple. Thus, it is easy to verify if models learned such rules by evaluating them on
numbers that are larger than the ones they were trained on. If successful, such a model would have
no problem correctly adding or subtracting arbitrarily long numbers

**прямой или обратный порядок предсказания**
The difference in accuracy is negligible between regular and inverse orders on interpolation tasks.
However, models trained and evaluated on the regular order show higher extrapolation accuracy
than those that use the inverse order

This result is perhaps surprising
since one would expect that the inverse order would be easier to learn.


**обучать с как минимум 50 цифрами**
Extrapolation is hardly achieved when trained on fewer than 50 digits, regardless of the model size



**нет смысла увеличивать обучающую выборку И на каком-то шаге обучения качество начинает падать, так что надо находить наилучший результат из промежуточных**




### второй 

python main.py --output_dir=.  --model_name_or_path=t5-base --operation=addition --orthography=10ebased --train_size=100 --val_size=10 --test_size=10 --min_digits_train=2  --max_digits_train=15 --min_digits_test=2 --max_digits_test=15 --base_number=10 --seed=1 --max_epochs=10 --check_val_every_n_epoch=2
 


larger models might perform better on data whose distribution i outside its training data distribution. поэтому не t5-base, а T5-3B

regular_order




## как генерируются тестовые данные 

-- см INVESTIGATING_THE_LIMITATIONS_OF_TRANSFORM_apr21 2 метода.


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
