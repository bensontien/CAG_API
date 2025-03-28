import os
import torch
import configparser
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, RepetitionPenaltyLogitsProcessor

config = configparser.ConfigParser()
config.read('config.ini')

app = FastAPI()

kv_cache = DynamicCache()
origin_len = 0
model = None
tokenizer = None
device = "cuda:0"

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

class QuestionRequest(BaseModel):
    query: str
    cache_name: str

def preprocess_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True
        )
    
    origin_len = cache.key_cache[0].shape[-2]
    clean_up(cache, origin_len)

    return cache

def write_kv_cache(kv: DynamicCache, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    """
    Write the KV Cache to a file.
    """
    torch.save(kv, path)

def read_kv_cache(path: str) -> DynamicCache | None:
    """
    Read the KV Cache from a file. If the cache file is invalid or empty, return None.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        kv = torch.load(path, weights_only=True)
        return kv
    else:
        return None

def clean_up(cache: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length. To avoid memory leak.
    """
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

def generate(model, input_ids: torch.Tensor, past_key_values, max_new_tokens: int = 2048, repetition_penalty: float = 1.15, temperature: float = 0.1) -> torch.Tensor:
    """Generate response"""
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids

    logits_processor = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :] / temperature
            logits = logits_processor(output_ids, logits)
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)

            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break

    return output_ids[:, input_ids.shape[-1]:]

def startup_event():
    """Initialize model and tokenizer"""
    global model, tokenizer, device
    model_name = config["model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    print(f"Loaded {model_name}.")

    docs = config["data"]["data_set"].replace(' ', '').split(',')
    for doc in docs:
        cache = set_kv_cache(f"/app/CAG/Data/{doc}.txt")
        write_kv_cache(cache, f"/app/CAG/Data/{doc}_kv_cache.pt")

   
def set_kv_cache(file_path: str):
    """Load document and build cache"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Please create a `{file_path}`")

    with open(file_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    system_prompt = f"""
    <|system|>
    You are an assistant who provides concise factual answers.
    <|user|>
    Context:
    {doc_text}
    Question:
    """.strip()

    cache = preprocess_kv_cache(model, tokenizer, system_prompt)
    
    print("KV cache built.")

    return cache

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    query = request.query
    cache_name = request.cache_name
    input_ids_q = tokenizer(query + "\n", return_tensors="pt").input_ids.to(device)
    kv_cache = read_kv_cache(f"/app/CAG/Data/{cache_name}_kv_cache.pt")
    gen_ids_q = generate(model, input_ids_q, kv_cache)
    answer = tokenizer.decode(gen_ids_q[0], skip_special_tokens=True).replace("Answer:", "").strip()
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    startup_event()
    uvicorn.run(app, host="0.0.0.0", port=59488)