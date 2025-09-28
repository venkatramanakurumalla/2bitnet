# %% CELL 1: SETUP & IMPORTS
print("🔧 BITNET-NSG INFERENCE - PRODUCTION READY")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import threading
from pathlib import Path
from einops import rearrange

# Environment detection
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Setup paths
if IN_COLAB:
    drive.mount('/content/drive')
    SAVE_PATH = "/content/drive/MyDrive/bitnet_nsg/"
else:
    SAVE_PATH = "./bitnet_nsg/"

MODEL_PATH = f"{SAVE_PATH}bitnet2b_final.pt"
TOKENIZER_MODEL_PATH = f"{SAVE_PATH}bitnet2b_en.model"
CONFIG_PATH = f"{SAVE_PATH}bitnet_nsg_config.json"

os.makedirs(SAVE_PATH, exist_ok=True)
print(f"✅ Setup complete! | Path: {SAVE_PATH}")

# Ensure consistent precision
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True

# %% CELL 2: OPTIMIZED MODEL ARCHITECTURE
print("🏗️ LOADING OPTIMIZED MODEL ARCHITECTURE")

class BitLinear2b(nn.Module):
    """Optimized 2-bit linear layer with robust shape handling"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize with smaller values for better stability
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None
        self.scale = nn.Parameter(torch.ones(out_features, dtype=torch.float32))

    def quantize_weight(self, w):
        return torch.round(torch.clamp(w, -2.0, 2.0) * 0.75 + 0.5) - 0.5

    def forward(self, x):
        w = self.quantize_weight(self.weight) if not self.training else \
            self.weight + (self.quantize_weight(self.weight) - self.weight).detach()

        # Robust broadcasting with explicit shape control
        scaled_weight = self.scale.view(-1, 1) * w
        return F.linear(x, scaled_weight, self.bias)

class BitNetBlock2b(nn.Module):
    """Memory-efficient transformer block"""
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_q = BitLinear2b(dim, dim)
        self.attn_k = BitLinear2b(dim, dim)
        self.attn_v = BitLinear2b(dim, dim)
        self.attn_out = BitLinear2b(dim, dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            BitLinear2b(dim, dim * ff_mult),
            nn.GELU(),
            BitLinear2b(dim * ff_mult, dim)
        )

    def forward(self, x):
        # Attention with memory optimization
        norm_x = self.norm1(x)
        q = rearrange(self.attn_q(norm_x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.attn_k(norm_x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.attn_v(norm_x), 'b n (h d) -> b h n d', h=self.heads)

        attn = (torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale).softmax(dim=-1)
        out = rearrange(torch.einsum('b h i j, b h j d -> b h i d', attn, v), 'b h n d -> b n (h d)')
        x = x + self.attn_out(out)
        x = x + self.ff(self.norm2(x))
        return x

class BitNetTransformer2b(nn.Module):
    """Complete BitNet transformer with tokenizer integration"""
    def __init__(self, num_tokens, dim, depth, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.blocks = nn.ModuleList([BitNetBlock2b(dim, dim_head, heads, ff_mult) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.to_logits = BitLinear2b(dim, num_tokens, bias=False)
        self.tokenizer = None

    def forward(self, x):
        x = self.emb(x)
        for block in self.blocks:
            x = block(x)
        return self.to_logits(self.norm(x))

print("✅ Model architecture ready!")

# %% CELL 3: ROBUST TOKENIZER LOADING
print("🔤 LOADING TOKENIZER...")

def setup_tokenizer():
    """Robust tokenizer loading with fallbacks"""
    try:
        import sentencepiece as spm
    except ImportError:
        print("📦 Installing sentencepiece...")
        os.system('pip install -q sentencepiece')
        import sentencepiece as spm

    # Find tokenizer file with multiple fallback strategies
    tokenizer_paths = [
        TOKENIZER_MODEL_PATH,
        f"{SAVE_PATH}tokenizer.model",
        *[os.path.join(SAVE_PATH, f) for f in os.listdir(SAVE_PATH) if f.endswith('.model')]
    ]

    for path in tokenizer_paths:
        if os.path.exists(path):
            try:
                sp_processor = spm.SentencePieceProcessor()
                if sp_processor.load(path):
                    print(f"✅ Tokenizer loaded from: {path}")
                    print(f"📊 Vocabulary size: {sp_processor.get_piece_size()}")
                    return sp_processor
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")
                continue

    raise FileNotFoundError("No valid tokenizer model found!")

class BitNetTokenizer:
    """Unified tokenizer interface"""
    def __init__(self, sp_processor):
        self.sp = sp_processor
        self.eos_id = 1  # SentencePiece default
        self.pad_id = 0

    def encode(self, text, return_tensors=None):
        ids = self.sp.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if not ids:
            return ""
        if isinstance(ids[0], list):
            ids = ids[0]
        return self.sp.decode(ids)

# Load tokenizer
sp_processor = setup_tokenizer()
tokenizer = BitNetTokenizer(sp_processor)
EOS_TOKEN_ID = tokenizer.eos_id
PAD_TOKEN_ID = tokenizer.pad_id

# %% CELL 4: MODEL LOADING WITH INTEGRITY CHECKS
print("🧠 LOADING TRAINED MODEL...")

def load_model_robust():
    """Load model with comprehensive error handling and validation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {device}")

    # Load configuration with smart defaults
    config = {
        "vocab_size": tokenizer.sp.get_piece_size(),
        "dim": 512,
        "depth": 8,
        "dim_head": 64,
        "heads": 8,
        "ff_mult": 4,
        "model_type": "BitNet-NSG-2b"
    }

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                saved_config = json.load(f)
            config.update(saved_config)
            print("✅ Configuration loaded from file")
        except Exception as e:
            print(f"⚠️ Config load failed, using defaults: {e}")

    # Validate vocab size consistency
    if config["vocab_size"] != tokenizer.sp.get_piece_size():
        print(f"⚠️ Vocab size mismatch! Config: {config['vocab_size']}, Tokenizer: {tokenizer.sp.get_piece_size()}")
        config["vocab_size"] = tokenizer.sp.get_piece_size()

    try:
        # Initialize model
        model = BitNetTransformer2b(
            num_tokens=config["vocab_size"],
            dim=config["dim"],
            depth=config["depth"],
            dim_head=config["dim_head"],
            heads=config["heads"],
            ff_mult=config["ff_mult"]
        )
        model.tokenizer = tokenizer

        # Load weights with multiple strategies
        model_files = [MODEL_PATH]
        if IN_COLAB:
            model_files.append(f"{SAVE_PATH}autosave_latest.pt")

        weights_loaded = False
        for model_path in model_files:
            if os.path.exists(model_path):
                try:
                    print(f"🔍 Attempting to load: {model_path}")
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)

                    weights_loaded = True
                    print(f"✅ Successfully loaded weights from: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Load failed for {model_path}: {e}")
                    continue

        if not weights_loaded:
            raise FileNotFoundError("No valid model checkpoint found")

        model = model.to(device).eval()

        # Verify model integrity
        test_input = torch.randint(0, config["vocab_size"], (1, 10), device=device)
        with torch.no_grad():
            test_output = model(test_input)
        assert test_output.shape == (1, 10, config["vocab_size"]), "Model output shape mismatch"

        print("🎉 MODEL LOADED SUCCESSFULLY WITH INTEGRITY VERIFICATION!")
        return model, device, config

    except Exception as e:
        print(f"❌ Critical error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

model, device, config = load_model_robust()

# %% CELL 5: OPTIMIZED GENERATION ENGINE (FIXED WITH PAD TOKEN BLOCKING)
print("⚡ INITIALIZING GENERATION ENGINE...")

def apply_top_p_filtering(logits, top_p):
    """Robust top-p filtering implementation"""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep at least one token
    if sorted_indices_to_remove.any():
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('inf')

    return logits

def apply_top_k_filtering(logits, top_k):
    """Efficient top-k filtering"""
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits

    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float('inf')
    return logits

def simple_generate(model, prompt, max_new_tokens=100, temperature=0.9, top_k=50, top_p=0.9,
                   repetition_penalty=1.0, stop_on_eos=True):
    """FIXED: High-performance generation with PAD token blocking"""
    model.eval()
    with torch.no_grad():
        # Encode input with proper dimensions
        input_ids = tokenizer.encode(prompt)
        if not isinstance(input_ids, list):
            input_ids = [input_ids]

        generated = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, seq_len]

        for _ in range(max_new_tokens):
            if generated.size(1) > 1024:
                generated = generated[:, -1024:]

            logits = model(generated)  # [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # 🔥 CRITICAL FIX: Block PAD token (ID 0) from being generated
            next_token_logits[PAD_TOKEN_ID] = -float('inf')

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if token_id != PAD_TOKEN_ID:  # Skip PAD token
                        next_token_logits[token_id] /= repetition_penalty

            # Apply sampling with proper dimension handling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                next_token_logits = apply_top_k_filtering(next_token_logits, top_k)
                next_token_logits = apply_top_p_filtering(next_token_logits, top_p)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # [1, 1]
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1]

            generated = torch.cat([generated, next_token], dim=1)  # [1, seq_len+1]

            if stop_on_eos and next_token.item() == EOS_TOKEN_ID:
                break

        return tokenizer.decode(generated[0].cpu().tolist())

# %% CELL 6: NEURO-SYMBOLIC COMPONENTS (OPTIMIZED & FIXED WITH PAD TOKEN BLOCKING)
print("🎭 LOADING NEURO-SYMBOLIC GUARDRAILS...")

class OptimizedNeuroSymbolicGenerator:
    """Memory-efficient neuro-symbolic generator"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_words = {"the", "and", "was", "were", "has", "had", "with", "said", "went",
                          "a", "an", "is", "in", "on", "at", "to", "for", "of", "by"}

    def apply_repetition_penalty(self, logits, generated_tokens, penalty=1.2):
        """Vectorized repetition penalty"""
        unique_tokens = torch.unique(generated_tokens)
        # Skip PAD token in penalty
        unique_tokens = unique_tokens[unique_tokens != PAD_TOKEN_ID]
        logits[unique_tokens] /= penalty
        return logits

    def enforce_narrative_diversity(self, token_ids, logits, diversity_threshold=0.3):
        """Efficient diversity enforcement"""
        if len(token_ids) <= 15:
            return logits

        try:
            text = self.tokenizer.decode(token_ids)
            words = [w.lower() for w in text.split() if len(w) > 2 and w.lower() not in self.stop_words]
            if not words:
                return logits

            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < diversity_threshold:
                logits += torch.randn_like(logits) * 0.2
        except:
            pass  # Fallback to original logits on decode error

        return logits

class OptimizedStoryGrammarValidator:
    """Fast story grammar validation with precomputed tokens"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state_triggers = {
            0: ['boy', 'girl', 'man', 'woman', 'king', 'queen', 'prince', 'princess', 'dog', 'cat'],
            1: ['ran', 'fought', 'found', 'lost', 'saw', 'wanted', 'needed', 'cried', 'laughed'],
            2: ['then', 'finally', 'suddenly', 'happily', 'together', 'saved', 'learned'],
            3: ['end', 'the end', 'happily ever after', 'goodbye', 'finished', 'home']
        }

        # Precompute token IDs
        self.state_tokens = {}
        for state, words in self.state_triggers.items():
            tokens = []
            for word in words:
                try:
                    ids = tokenizer.encode(word)
                    if isinstance(ids, list) and len(ids) == 1:
                        tokens.append(ids[0])
                except:
                    continue
            self.state_tokens[state] = set(tokens)

    def get_current_state(self, token_ids):
        """Fast state detection"""
        tokens = token_ids[-20:] if len(token_ids) > 20 else token_ids

        # Check for ending tokens
        if any(tid in self.state_tokens[3] for tid in tokens[-5:]):
            return 3
        elif any(tid in self.state_tokens[2] for tid in tokens[-10:]):
            return 2
        elif any(tid in self.state_tokens[1] for tid in tokens[-15:]):
            return 1
        elif len(token_ids) < 10:
            return 0
        return 1  # Default to action state

    def validate_logits(self, current_tokens, logits):
        """Efficient grammar validation"""
        current_state = self.get_current_state(current_tokens)
        mask = torch.ones_like(logits)

        # Apply state-based masking
        if current_state == 0:  # Beginning - encourage character tokens
            for tid in self.state_tokens[0]:
                if tid < len(logits):
                    mask[tid] *= 1.5
        elif current_state == 3:  # Ending - discourage non-ending tokens
            for i in range(len(logits)):
                if i not in self.state_tokens[3]:
                    mask[i] *= 0.8

        return logits * mask

def neuro_symbolic_generate(model, prompt, max_new_tokens=100, temperature=0.9, top_k=50, top_p=0.9,
                          repetition_penalty=1.2, diversity_threshold=0.3, use_grammar=True):
    """FIXED: Full neuro-symbolic generation with PAD token blocking"""
    model.eval()
    nsg = OptimizedNeuroSymbolicGenerator(tokenizer)
    grammar_validator = OptimizedStoryGrammarValidator(tokenizer) if use_grammar else None

    with torch.no_grad():
        # Encode input with proper dimension handling
        input_ids = tokenizer.encode(prompt)
        if not isinstance(input_ids, list):
            input_ids = [input_ids]

        generated = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, seq_len]

        for _ in range(max_new_tokens):
            if generated.size(1) > 1024:
                generated = generated[:, -1024:]

            logits = model(generated)  # [1, seq_len, vocab_size]
            next_logits = logits[0, -1, :].clone()  # [vocab_size]

            # 🔥 CRITICAL FIX: Block PAD token (ID 0) from being generated
            next_logits[PAD_TOKEN_ID] = -float('inf')

            # Apply neuro-symbolic constraints
            if repetition_penalty != 1.0:
                next_logits = nsg.apply_repetition_penalty(next_logits, generated[0], repetition_penalty)

            if grammar_validator:
                next_logits = grammar_validator.validate_logits(generated[0].tolist(), next_logits)

            next_logits = nsg.enforce_narrative_diversity(generated[0].tolist(), next_logits, diversity_threshold)

            # Apply sampling with proper dimension handling
            if temperature > 0:
                next_logits = next_logits / temperature
                next_logits = apply_top_k_filtering(next_logits, top_k)
                next_logits = apply_top_p_filtering(next_logits, top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # [1, 1]
            else:
                next_token = torch.argmax(next_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1]

            # Concatenate along sequence dimension
            generated = torch.cat([generated, next_token], dim=1)  # [1, seq_len+1]

            if next_token.item() == EOS_TOKEN_ID:
                break

        return tokenizer.decode(generated[0].cpu().tolist())

print("✅ Neuro-symbolic components ready!")

# %% CELL 7: PARAMETER OPTIMIZATION
print("🎛️ OPTIMIZING GENERATION PARAMETERS...")

def find_optimal_parameters():
    """Find the best parameters for creative generation"""

    # Test prompts that encourage longer responses
    test_prompts = [
        "Once upon a time there was",
        "The little dog ran",
        "In a magical forest"
    ]

    # Parameter combinations to test
    param_combinations = [
        {"temperature": 1.3, "top_k": 80, "top_p": 0.92, "repetition_penalty": 1.05, "name": "Creative"},
        {"temperature": 1.1, "top_k": 60, "top_p": 0.90, "repetition_penalty": 1.1, "name": "Balanced"},
        {"temperature": 1.5, "top_k": 100, "top_p": 0.88, "repetition_penalty": 1.0, "name": "Very Creative"},
        {"temperature": 0.9, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.15, "name": "Conservative"},
    ]

    best_params = None
    best_score = 0

    print("🧪 Testing parameter combinations...")

    for params in param_combinations:
        total_length = 0
        success_count = 0

        for prompt in test_prompts:
            try:
                result = simple_generate(
                    model, prompt,
                    max_new_tokens=80,
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"]
                )

                # Score based on generation length beyond prompt
                gen_length = len(result) - len(prompt)
                if gen_length > 10:  # At least 10 new characters
                    success_count += 1
                    total_length += gen_length

            except Exception as e:
                continue

        score = success_count * total_length
        print(f"   {params['name']}: Score {score} (length: {total_length})")

        if score > best_score:
            best_score = score
            best_params = params

    if best_params:
        print(f"🎯 Best parameters: {best_params['name']}")
        # Remove the name for actual generation use
        best_params = {k: v for k, v in best_params.items() if k != 'name'}

    return best_params if best_params else {"temperature": 1.3, "top_k": 80, "top_p": 0.92, "repetition_penalty": 1.05}

# Find optimal parameters
optimal_params = find_optimal_parameters()
print(f"✅ Optimal parameters: {optimal_params}")

# %% CELL 8: ENHANCED INTERACTIVE INTERFACE
def progress_indicator():
    """Show animated progress dots during generation"""
    stop_event = threading.Event()

    def animate():
        dots = 0
        while not stop_event.is_set():
            print(".", end="", flush=True)
            dots += 1
            time.sleep(1)
            if dots >= 3:
                print("\b\b\b   \b\b\b", end="", flush=True)
                dots = 0

    thread = threading.Thread(target=animate)
    thread.daemon = True
    thread.start()
    return stop_event

def enhanced_interactive_generation():
    """Enhanced interactive generation with optimized parameters"""
    if model is None:
        print("❌ Model not available for generation")
        return

    # Default parameters (optimized for creativity)
    params = {
        'max_new_tokens': 150,  # Increased for longer stories
        'temperature': 1.3,     # Higher for more creativity
        'top_k': 80,
        'top_p': 0.92,
        'repetition_penalty': 1.05,  # Lower to allow some repetition
        'neuro_symbolic': True,
    }

    # Override with optimal parameters if found
    params.update(optimal_params)

    print("\n" + "="*70)
    print("🤖 BITNET-NSG ENHANCED INFERENCE CONSOLE")
    print("="*70)
    print("💡 TIPS: Use complete story starters for best results!")
    print("   Good: 'Once upon a time there was a little dog who'")
    print("   Good: 'The brave knight entered the dark castle and'")
    print("   Good: 'In a magical forest, a girl carried a lamp and'")
    print("\n💡 COMMANDS:")
    print("   /params    - Adjust generation parameters")
    print("   /mode      - Toggle neuro-symbolic mode")
    print("   /creative  - More creative settings")
    print("   /safe      - More conservative settings")
    print("   /examples  - Show example prompts")
    print("   /benchmark - Run speed benchmark")
    print("   /quit      - Exit gracefully")
    print("="*70)

    def display_params():
        print("\n📊 Current Parameters:")
        for k, v in params.items():
            print(f"   {k}: {v}")

    def show_examples():
        examples = [
            "Once upon a time there was a little dog who",
            "The brave knight entered the dark castle and",
            "In a magical forest, a girl carried a lamp and",
            "The sun was rising when suddenly",
            "Tom was wonderful because he always"
        ]
        print("\n💡 EXAMPLE PROMPTS:")
        for i, example in enumerate(examples, 1):
            print(f"   {i}. '{example}'")

    while True:
        try:
            user_input = input("\n🎯 Prompt> ").strip()

            if not user_input:
                continue
            elif user_input.lower() == '/quit':
                break
            elif user_input.lower() == '/params':
                display_params()
                try:
                    print("\n📝 Enter new values (press Enter to keep current):")
                    params['max_new_tokens'] = int(input(f"Max tokens [{params['max_new_tokens']}]: ") or params['max_new_tokens'])
                    params['temperature'] = float(input(f"Temperature [{params['temperature']}]: ") or params['temperature'])
                    params['top_k'] = int(input(f"Top-k [{params['top_k']}]: ") or params['top_k'])
                    params['top_p'] = float(input(f"Top-p [{params['top_p']}]: ") or params['top_p'])
                    params['repetition_penalty'] = float(input(f"Rep penalty [{params['repetition_penalty']}]: ") or params['repetition_penalty'])
                    print("✅ Parameters updated!")
                except ValueError:
                    print("❌ Invalid input - parameters unchanged")
                continue
            elif user_input.lower() == '/creative':
                params.update({'temperature': 1.5, 'top_p': 0.88, 'repetition_penalty': 1.0})
                print("🎨 Creative mode activated! (higher temperature, more randomness)")
                continue
            elif user_input.lower() == '/safe':
                params.update({'temperature': 0.8, 'top_p': 0.95, 'repetition_penalty': 1.2})
                print("🛡️ Safe mode activated! (more conservative, less random)")
                continue
            elif user_input.lower() == '/mode':
                params['neuro_symbolic'] = not params['neuro_symbolic']
                mode = "Neuro-Symbolic" if params['neuro_symbolic'] else "Simple"
                print(f"🔄 Switched to {mode} mode")
                continue
            elif user_input.lower() == '/examples':
                show_examples()
                continue
            elif user_input.lower() == '/benchmark':
                print("⏱️ Running benchmark...")
                start = time.time()
                test_result = simple_generate(model, "The quick brown fox", max_new_tokens=50, temperature=1.0)
                elapsed = time.time() - start
                tokens = len(tokenizer.encode(test_result))
                print(f"📈 Speed: {tokens/elapsed:.1f} tokens/sec | {elapsed:.2f}s total")
                continue

            # Generate response with progress indicator
            print("🔮 Generating", end="", flush=True)
            start_time = time.time()

            progress = progress_indicator()

            try:
                if params['neuro_symbolic']:
                    result = neuro_symbolic_generate(
                        model, user_input,
                        max_new_tokens=params['max_new_tokens'],
                        temperature=params['temperature'],
                        top_k=params['top_k'],
                        top_p=params['top_p'],
                        repetition_penalty=params['repetition_penalty']
                    )
                else:
                    result = simple_generate(
                        model, user_input,
                        max_new_tokens=params['max_new_tokens'],
                        temperature=params['temperature'],
                        top_k=params['top_k'],
                        top_p=params['top_p'],
                        repetition_penalty=params['repetition_penalty']
                    )
            finally:
                progress.set()
                time.sleep(0.1)  # Let the animation thread clean up
                print("", end="\r", flush=True)  # Clear the progress line

            gen_time = time.time() - start_time
            prompt_tokens = len(tokenizer.encode(user_input))
            result_tokens = len(tokenizer.encode(result))
            new_tokens = result_tokens - prompt_tokens

            print(f"✅ Generated {new_tokens} new tokens in {gen_time:.2f}s ({new_tokens/gen_time:.1f} t/s)")
            print(f"\n📖 STORY:")
            print("─" * 60)
            print(result)
            print("─" * 60)
            print(f"\n💡 Prompt: '{user_input}'")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using BitNet-NSG!")
            break
        except Exception as e:
            print(f"\n❌ Generation error: {e}")
            import traceback
            traceback.print_exc()

# %% CELL 9: QUICK DEMONSTRATION
print("🚀 DEMONSTRATING OPTIMIZED GENERATION...")

def quick_demo():
    """Show what the model can do with optimized parameters"""
    if model is None:
        return

    demo_prompts = [
        "Once upon a time there was a little dog who",
        "The brave knight entered the dark castle and",
        "In a magical forest, a girl carried a lamp and"
    ]

    print("\n🌟 QUICK DEMONSTRATION:")
    print("=" * 60)

    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        try:
            result = neuro_symbolic_generate(
                model, prompt,
                max_new_tokens=80,
                temperature=1.3,
                top_k=80,
                top_p=0.92,
                repetition_penalty=1.05
            )
            # Extract only the generated part (after prompt)
            generated_part = result[len(prompt):].strip()
            if generated_part:
                print(f"   Generated: {generated_part}")
            else:
                print("   (Short response - try different parameters)")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("💡 Try these prompts in interactive mode for longer stories!")
    print("   Use '/creative' for more adventurous generations")
    print("   Use '/safe' for more conservative generations")

# Run demonstration
quick_demo()

# %% CELL 10: MAIN EXECUTION
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 BITNET-NSG INFERENCE SYSTEM READY")
    print("="*70)

    if model is not None:
        # Display system information
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model: {config.get('model_type', 'BitNet-NSG-2b')}")
        print(f"📈 Parameters: {total_params:,}")
        print(f"🎯 Vocabulary: {config['vocab_size']:,} tokens")
        print(f"🏗️ Architecture: {config['dim']}d, {config['depth']} layers")
        print(f"🧠 Device: {device.upper()}")
        print(f"🎛️ Generation: Creative Mode (temp: {optimal_params.get('temperature', 1.3)})")
        print("="*70)

        # Start enhanced interactive mode
        enhanced_interactive_generation()
    else:
        print("❌ FATAL: Model failed to load - please check your checkpoint files")
        print(f"🔍 Expected model at: {MODEL_PATH}")
        print(f"🔍 Expected tokenizer at: {TOKENIZER_MODEL_PATH}")

    print("\n🎉 BITNET-NSG INFERENCE SESSION COMPLETE!")
