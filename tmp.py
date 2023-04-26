from transformers import AutoTokenizer, AutoModelForCausalLM

###
from few_shot_with_py.prompts_formatting import generate_prompt


SUM_PROMPT = generate_prompt()
question = SUM_PROMPT.format(a=1, b=2)
print(question)
######

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")

# text = "def sum(a: int, b:int) -> int:"
text = question
input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
