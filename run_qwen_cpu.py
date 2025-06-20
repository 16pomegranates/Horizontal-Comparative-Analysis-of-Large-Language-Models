from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

# Model path
model_name = "/mnt/data/Qwen-7B-Chat"  # Local model path

# Input prompts
prompts = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "他知道我知道你知道他不知道吗？",
    "明明明明明白白白喜欢他，可她就是不说",
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。"
]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"  # Automatically selects Float32/Float16 (based on model config)
).eval()

# Create TextStreamer
streamer = TextStreamer(tokenizer)

# Generate answers for each prompt
for i, prompt in enumerate(prompts, start=1):
    print(f"问题 {i}: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
    print(f"回答 {i}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print("=" * 80)
