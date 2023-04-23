# huawei_test_task

## вопросы

как генерируются тестовые данные -- см INVESTIGATING_THE_LIMITATIONS_OF_TRANSFORM_apr21 2 метода.

### Задача
На экзамене по AGI в институте, где учится Петя, разрешается пользоваться 
калькулятором для вычислений, и большими языковыми моделями (не 
более 4B параметров) для ответа на любые вопросы. У Пети сломался 
калькулятор. Помогите ему адаптировать языковую модель так, чтобы на 
экзамене у него не было проблем со сложением длинных чисел.


### Комментарий
Необходимо обучить языковую модель решать примеры на сложение как 
можно более длинных чисел. Качество будет измеряться на случайно 
сгенерированном наборе чисел разной длины. 
Необходимо предоставить код, а также технический отчет, содержащий 
описание метода и используемых данных, оценку качества, и краткий обзор 
литературы по данной теме

Word_math_problems_april_23 -- слишком большая для нашей задачи

Word_math_problems_april_23 -> PAL: Program-aided Language Models (который вышел раньше)

Оттуда their performance falls dramatically when dealing with complex arithmetic (Hendrycks et al., 2021; Madaan & Yazdanbakhsh,
2022) or large numbers (Nogueira et al., 2021; Qian et al., 2022).

Вот эта статья про арифметику
https://arxiv.org/abs/2102.13019 (
код https://github.com/castorini/transformers-arithmetic
INVESTIGATING_THE_LIMITATIONS_OF_TRANSFORM_apr21

## PAL
http://reasonwithpal.com .

Главное -- переводить вычисления в код.

**В PAL они заметили, что в датасетах маленькие числа и решили их увеличить до 7-ми значных**

GSM-HARD LLMs can perform simple calculations with
small numbers. However, Madaan & Yazdanbakhsh (2022)
found that 50% of the numbers in the popular GSM8K
dataset of math reasoning problems are integers between 0
and 8. This raises the question of whether LLMs can generalize to larger and non-integer numbers? We constructed a
harder version of GSM8K, which we call GSM-HARD, by replacing the numbers in the questions of GSM8K with larger
numbers. Specifically, one of the numbers in a question
was replaced with a random integer of up to 7 digits. More
details regarding the this new dataset are provided in H.1.

**Large Numbers or Incorrect Reasoning**
Они пытались понять ,почему ошибаются LLMs -- проблем с переводом или больщими числами

the primary failure mode is the inability to perform arithmetic accurately

ЭТО ОЧЕНЬ ВАЖНО!!!!!!!!!!!!!!
ТО ЕСТЬ НАДО НЕ ОСТАВЛЯТЬ ВЫЧИСЛЕНИЯ МОДЕЛИ, А В ПИТОН

**На более простых моделях?**

PAL with COT при более простой базе. PAL -- всегда лучше

**Это именно засчет использования  интерпретатора или из-за того, что больше данных prompts (впрямую считай)**
интерпретатор


**ссылка на упрощение**
Several prior works have
equipped neural models with specialized modules. For example, Cobbe et al. (
2021) employ a calculator for arithmetic operations as a post hoc processing,



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


## текущая идея
использовать интерпретатор из PAL, носпецифицировать его именно на суммирование, используя идеи investigationg




## пошла искать, кто ссылается на investigationg

https://ojs.aaai.org/index.php/AAAI/article/view/20841

https://arxiv.org/abs/2103.13136

2 подхода -- пытаться лучше представить числа или в интерпретатор 

еще про CoT сказать но вот есть еще и такое



моделей тех нет, взяла лучшее 
https://platform.openai.com/docs/model-index-for-researchers

еще работает gpt-3.5-turbo


а  Declarative solutionsб описанные в word_math_problems -- подходят для более сложных


Здравствуйте. Появились вопросы по поводу того, в каком формате будут предоставляться входные данные.

1. В каждом примере будут складываться только 2 числа или произвольное их количество?
2. Складываемые числа -- любые или есть какие-то ограничения? (например, только целые, или только целые неотрицательные)
3. Как именно требуется передавать задачу в модель?  
	а) Будут даны только числа, а как их передавать в модель -- на мое усмтрение.
	б) Будет сразу дано некоторое предложение на естественном языке, содержащее задачу о суммировании чисел, которое нужно именно в таком виде (без изменений) передать в модель



о том, что использую в первом способе -- вот, из word_math_problems:
можно сказать что вот видела интерпретатор, и как тут декларативный
Few-shot prompting is a technique that uses LLMs to solve a task by providing the LLMs with
a few demonstrations of the task as part of the input at inference time [1].




python main.py --output_dir=.  --model_name_or_path=t5-base --operation=addition --orthography=10ebased --train_size=100 --val_size=10 --test_size=10 --min_digits_train=2  --max_digits_train=15 --min_digits_test=2 --max_digits_test=15 --base_number=10 --seed=1 --max_epochs=10 --check_val_every_n_epoch=2
 
 
все ограничено разрядной сеткой, так что давайте лучше разделять на части

начнем с первого, что приходит в голову

3 варианта 
Питон без итерации
Питон с итерацией
Без питона







