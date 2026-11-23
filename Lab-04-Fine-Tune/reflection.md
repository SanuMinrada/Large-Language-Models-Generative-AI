# Week 4 Reflection — Fine-Tuning an LLM with LoRA

## What I learned

This lab made fine-tuning feel like a real, hands-on process instead of just a concept. I picked a
small model (`SmolLM2-360M-Instruct`) and made up my own 32-question dataset for a fake computer
repair shop, then fine-tuned it with LoRA on just my CPU. It only took about 5 minutes, which
surprised me — I expected fine-tuning to need a GPU no matter what, but LoRA only trains a small
set of extra parameters (about 1.2% of the model), so it's light enough for a small model like
this one. The loss dropped steadily from 3.18 to 2.18, and I could actually see the model's answers
change in tone afterwards — it started answering in character as "ByteFix" instead of giving
generic AI-assistant disclaimers.

## What was challenging

I found a real bug while comparing the models: if you load the fine-tuned model by wrapping the
*same* base model object with `PeftModel.from_pretrained`, it mutates that object in place. So if
you then reuse that same object thinking it's still the "original" model, you're actually
comparing the fine-tuned model against itself. I had to load two completely separate model
instances to compare them fairly. It also took me a while to accept an honest result: the
fine-tuned model doesn't reliably repeat the exact prices I trained it on — it sometimes makes up
numbers instead of using the real LKR figures. With only 32 examples and 5 epochs, that's a fair
result, not something to hide.

## Testing seen vs. unseen questions

Testing with questions that were and weren't in the training data made the comparison much more
convincing. For a seen question like the refurbished laptop warranty, the fine-tuned model gave a
short, confident answer instead of the base model's long hedge. For an unseen question about a
loyalty program, the fine-tuned model still answered in the ByteFix "voice" even though that exact
fact was never in the training data, which showed it had picked up on the style, not just
memorised answers.

## What I'd try next

- Train for more epochs or with a larger dataset to see if exact fact recall improves.
- Try a slightly bigger model like `Qwen2.5-0.5B-Instruct` and compare the results.
- Run the same experiment on a GPU (Colab) to see how much faster it trains.
