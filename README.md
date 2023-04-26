# huawei_test_task


# Problem description

Adapt a large language model (not more than 4B parameters) to solve the problem of **addition of two long integers** (as long numbers as possible).
The quality will be measured on a randomly generated set of numbers of different lengths.


### Original task in Russian

#### Задание
На экзамене по AGI в институте, где учится Петя, разрешается пользоваться 
калькулятором для вычислений, и большими языковыми моделями (не 
более 4B параметров) для ответа на любые вопросы. У Пети сломался 
калькулятор. Помогите ему адаптировать языковую модель так, чтобы на 
экзамене у него не было проблем со сложением длинных чисел.


#### Комментарий
Необходимо обучить языковую модель решать примеры на сложение как 
можно более длинных чисел. Качество будет измеряться на случайно 
сгенерированном наборе чисел разной длины. 
Необходимо предоставить код, а также технический отчет, содержащий 
описание метода и используемых данных, оценку качества, и краткий обзор 
литературы по данной теме


# Literature review: 


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


## Review



The first possible direction of solving the problem of addition of two long integers as one of the mathematical problems is to 'teach' language models the process of reasoning.

Recently, LLMs have shown impressive success on a wide range of tasks, including mathematical, using this approach.

For example, paper [[1]](#1) demonstrates that by fine-tuning an autoregressive language model (GPT-Neo) on appropriately structured step-by-step demonstrations, it is possible to teach it to execute 
a mathematical task that has previously proved difficult for Transformers – longhand modulo  operations – with a relatively small number of examples. 
 Their demonstrations are presented in a widely used **chain-of-thought (COT)** form.


Nevertheless, while LLMs can 'themselves' perform simple arithmetic operations, their performance falls dramatically when dealing with large numbers ([[6]](#6)). In fact, 
even when fine-tuning a PaLM-based model on 164B tokens of explicit mathematical content, one of its two
most common failures is “incorrect calculation” [[7]](#7).

That's why it is worth considering a slightly different approach.
Papers [[2]](#2) and [[3]](#3) go in approximately in the same direction but use few-shot prompting in some kind of **algorithmic form** ([[4]](#4), [5]](#5)).

- Proposed in [[3]](#3) approach uses the LLM to read natural language problems and generate programs as the intermediate
reasoning steps, but offloads the solution step to a runtime (**Python interpreter**)

- Proposed in [[2]](#2) approach combines an LLM that can incrementally formalize word problems as a set of variables and 
equations with an external **symbolic solver** that can solve them


** My first solution was mostly inspired by and based on paper [[3]](#3)**. Here are some reasons:

1. It has been shown in [[2]](#2) that their approach is more effective for more difficult problems that require declarative reasoning while the 
results on simple tasks like ours (addition of two long integers) [[2]](#2) and [[3]](#3) show comparable results.


2. Since python 3 has no more limit to value of integers (https://docs.python.org/3/whatsnew/3.0.html#integers), almost all limits for the length of
 numbers to sum.
 

Here are some more interesting insights from [[3]](#3)

1. Large Numbers or Incorrect Reasoning? 

Autrhors show that the primary failure mode during working with large numbers is the inability to perform such arithmetic accurately, not the wrong generated solution steps

So the main thing to focus on is performing arithmetic accurately which is esay to do using python.

2. Is PAL better because of the Python prompt or because of the interpreter? 

Authors experimented with generating Python code, while requiring the neural LM to “execute” it as well, without using an interpreter. 
They created prompts that are similar to PAL’s, except that they do include the final answer.
This resulted in a much lower accuracy results compared to original approach and other models.
These results reinforced thair hypothesis that the main benefit of PAL comes from the synergy with the interpreter, and not only from having a better prompt.


3. PAL can work with weaker models, while its benefit over chain-of-thought scales elegantly to stronger models as well.




# Solution 1. few-shot with python interpreter (based on [[3]](#3))

Осталось найти модель с небольшим числом параемтров и описать.


Few-shot prompting leverages the strength of large-language
models to solve a task with a set of k examples that are provided as part of the test-time input ([4]](#4), [5]](#5)), where k is usually a number in the low single digits. These input-output
examples {(xi, yi)}, i=1;k are concatenated in a prompt p ≡ (x1 · y1) || (x2 · y2) || ... || (xk · yk). where “·” denotes
the concatenation of an input and output, and “||” indicate
the concatenation of different examples. During inference,
a test instance xtest is appended to the prompt, and p k xtest
is passed to the model which attempts to complete p k xtest,
and thereby generate an answer ytest. Note that such fewshot prompting does not modify the underlying LLM.

ограничение -- по max_tokens

text-davinci-003
MAX TOKENS
4,097 tokens

https://platform.openai.com/docs/models/gpt-4


python main.py --a=10000000001 --b=9


## Base model
In the original paper authors mostly used CODEX (code-davinci-002) as a backend LLM. However Codex models are now deprecated (https://platform.openai.com/docs/models/codex).
They were descendants of our GPT-3 models that would understand and generate code. 



моделей тех нет, взяла лучшее 
https://platform.openai.com/docs/model-index-for-researchers

еще работает gpt-3.5-turbo

в оригинале использовался davinci code, у меня -- текст


Хотя авторы говорили, что так делать не надо

Does PAL work with LMs of natural language? We
also experimented with PAL using the text-davinci
series. Figure 8 shows the following interesting results: when the base LM’s “code modeling ability” is
weak (using text-davinci-001), COT performs better
than PAL. However, once the LM’s code modeling ability is sufficiently high (using text-davinci-002 and
text-davinci-003), PAL outperforms COT, and PAL
text-davinci-003 performs almost as PAL code-davinci-002.
This shows that PAL is not limited to LMs of code, but it
can work with LMs that were mainly trained for natural
language, if they have a sufficiently high coding ability.

### альтернативы


 https://arxiv.org/pdf/1910.00577v1.pdf -- 15m, C#, java
 
 
 gpt 2
 

## возможные проблемы -- когда число циклическое, то модель скорее повторять начинает. так было когда комментарий писала как в оригинальной статье
 

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
 

в питоне есть еще majority at





## как генерируются тестовые данные 

-- см INVESTIGATING_THE_LIMITATIONS_OF_TRANSFORM_apr21 2 метода.

