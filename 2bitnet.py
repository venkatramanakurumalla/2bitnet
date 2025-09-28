
# =====================================================
# 🌟 BITNET-NSG 2b AUTO-TRAINER — COLAB & KAGGLE READY
# ✅ Auto GPU | Auto Save | Auto Install | Auto Upload to Drive
# ✅ Crash-proof: resumes from last good checkpoint
# ✅ AII — AI INSPECTION: Generates sample after every checkpoint
# ✅ Safe atomic saving — prevents file corruption

# =====================================================

import os
import sys
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import zipfile
import json
from tqdm import tqdm
import time
from datetime import datetime

# =====================================================
# 🔧 INSTALL DEPENDENCIES (Silent)
# =====================================================

print("🔧 INSTALLING DEPENDENCIES...")

try:
    import sentencepiece as spm
except:
    !pip install -q sentencepiece einops tqdm
    import sentencepiece as spm

try:
    from tqdm import tqdm
except:
    !pip install -q tqdm

# Detect environment
IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle/working')

print(f"🌍 Environment: {'Google Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'Local'}")

# Setup save path
if IN_COLAB:
    from google.colab import drive
    print("☁️  MOUNTING GOOGLE DRIVE...")
    drive.mount('/content/drive')
    SAVE_PATH = "/content/drive/MyDrive/bitnet_nsg/"
    os.makedirs(SAVE_PATH, exist_ok=True)
elif IN_KAGGLE:
    SAVE_PATH = "/kaggle/working/"
else:
    SAVE_PATH = "./"

print(f"💾 Models and samples will be saved to: {SAVE_PATH}")

# =====================================================
# 🔧 SAFE SAVE FUNCTION — PREVENTS CORRUPTION
# =====================================================

def safe_save_checkpoint(state_dict, path):
    """Save checkpoint safely — write to temp file first, then rename."""
    temp_path = path + ".tmp"
    torch.save(state_dict, temp_path)
    os.replace(temp_path, path)  # Atomic on most systems
    print(f"💾 Safely saved checkpoint to: {path}")

# =====================================================
# 🔧 MODEL DEFINITION — 2-BIT BITNET
# =====================================================

class BitLinear2b(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = nn.Parameter(torch.ones(out_features))

    def quantize_weight(self, w):
        w = w.clamp(-2, 2)
        w = w * 0.75
        return torch.round(w + 0.5) - 0.5

    def forward(self, x):
        if self.training:
            w_quant = self.quantize_weight(self.weight)
            w = self.weight + (w_quant - self.weight).detach()
        else:
            w = self.quantize_weight(self.weight)
        return F.linear(x, self.scale.unsqueeze(1) * w, self.bias)

class BitNetBlock2b(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_q = BitLinear2b(dim, dim)
        self.attn_k = BitLinear2b(dim, dim)
        self.attn_v = BitLinear2b(dim, dim)
        self.attn_out = BitLinear2b(dim, dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            BitLinear2b(dim, dim * ff_mult),
            nn.GELU(),
            BitLinear2b(dim * ff_mult, dim)
        )

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        qkv = [self.attn_q(self.norm1(x)), self.attn_k(self.norm1(x)), self.attn_v(self.norm1(x))]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        x = x + self.attn_out(out)
        x = x + self.ff(self.norm2(x))
        return x

class BitNetTransformer2b(nn.Module):
    def __init__(self, num_tokens, dim, depth, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.blocks = nn.ModuleList([BitNetBlock2b(dim, dim_head, heads, ff_mult) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.to_logits = BitLinear2b(dim, num_tokens, bias=False)
        self.tokenizer = None  # Will be set externally

    def forward(self, x):
        x = self.emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.to_logits(x)

# =====================================================
# 🎭 NEURO-SYMBOLIC GUARDRAILS
# =====================================================

class NeuroSymbolicGenerator:
    def __init__(self, tokenizer, min_new_concepts=3, penalty_strength=2.0, diversity_threshold=0.3):
        self.tokenizer = tokenizer
        self.min_new_concepts = min_new_concepts
        self.penalty_strength = penalty_strength
        self.diversity_threshold = diversity_threshold

    def apply_repetition_penalty(self, logits, generated_tokens, penalty=None):
        penalty = penalty or self.penalty_strength
        for token in set(generated_tokens.tolist()):
            logits[token] -= penalty
            if logits[token] < -10:
                logits[token] = -10
        return logits

    def enforce_narrative_diversity(self, token_ids, logits, force_temp=0.9):
        if len(token_ids) > 15:
            text = self.tokenizer.decode(token_ids)
            words = [w for w in text.lower().split()
                    if len(w) > 2 and w not in {"the", "and", "was", "were", "has", "had", "with", "said", "went", "a", "an", "is"}]
            unique_ratio = len(set(words)) / max(1, len(words))

            if unique_ratio < self.diversity_threshold:
                logits += torch.randn_like(logits) * 0.3
                probs = torch.softmax(logits / force_temp, dim=-1)
                return torch.log(probs + 1e-8)
        return logits

# =====================================================
# 📖 STORY GRAMMAR VALIDATOR
# =====================================================

class StoryGrammarValidator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.states = {'BEGIN': 0, 'CHARACTER': 1, 'ACTION': 2, 'RESOLUTION': 3, 'END': 4}

        self.triggers = {
            'CHARACTER': ['boy', 'girl', 'man', 'woman', 'king', 'queen', 'prince', 'princess',
                         'dog', 'cat', 'dragon', 'knight', 'wizard', 'witch', 'robot', 'child', 'farmer', 'elf', 'baby', 'pirate'],
            'ACTION': ['ran', 'fought', 'found', 'lost', 'saw', 'wanted', 'needed', 'cried',
                      'laughed', 'built', 'destroyed', 'escaped', 'discovered', 'asked', 'went', 'jumped', 'flew', 'climbed', 'sailed'],
            'RESOLUTION': ['then', 'finally', 'suddenly', 'happily', 'together', 'saved',
                          'learned', 'realized', 'returned', 'fixed', 'won', 'helped', 'gave', 'shared', 'decided', 'understood'],
            'END': ['end', 'the end', 'happily ever after', 'goodbye', 'finished', 'done', 'sleep', 'smiled', 'home', 'peace', 'forever']
        }
        self.token_triggers = self._precompute_token_triggers()

    def _precompute_token_triggers(self):
        token_triggers = {}
        for state, words in self.triggers.items():
            token_ids = []
            for word in words:
                ids = self.tokenizer.encode(word)
                if isinstance(ids, list) and len(ids) == 1:
                    token_ids.append(ids[0])
            token_triggers[state] = set(token_ids)
        return token_triggers

    def get_current_state(self, token_ids):
        tokens = token_ids.tolist() if torch.is_tensor(token_ids) else token_ids
        if len(tokens) > 40 and any(tid in self.token_triggers['END'] for tid in tokens[-5:]):
            return self.states['END']
        elif any(tid in self.token_triggers['RESOLUTION'] for tid in tokens[-10:]):
            return self.states['RESOLUTION']
        elif any(tid in self.token_triggers['ACTION'] for tid in tokens[-15:]):
            return self.states['ACTION']
        elif any(tid in self.token_triggers['CHARACTER'] for tid in tokens[:10]):
            return self.states['CHARACTER']
        else:
            return self.states['BEGIN']

    def should_allow_transition(self, current_state, candidate_token, token_ids):
        transition_rules = {
            0: [0,1],
            1: [1,2],
            2: [2,3],
            3: [3,4],
            4: [4]
        }
        triggered_state = 0
        for state_name, state_id in self.states.items():
            if candidate_token in self.token_triggers.get(state_name, set()):
                triggered_state = state_id
                break

        if current_state in [0,1,2] and triggered_state == 4:
            return False
        if current_state == 3 and len(token_ids) < 50:
            if triggered_state == 4:
                return False

        return triggered_state in transition_rules[current_state]

    def validate_and_score(self, current_tokens, candidate_logits):
        current_state = self.get_current_state(current_tokens)
        mask = torch.ones_like(candidate_logits)

        for i in range(len(candidate_logits)):
            if not self.should_allow_transition(current_state, i, current_tokens):
                mask[i] = 0.05

        bonus_multiplier = 1.0
        if current_state == 3:
            bonus_multiplier = min(2.0, len(current_tokens) / 60.0)
        elif current_state == 2:
            bonus_multiplier = min(1.5, len(current_tokens) / 40.0)

        progress_bonuses = {0: self.token_triggers['CHARACTER'], 1: self.token_triggers['ACTION'],
                           2: self.token_triggers['RESOLUTION'], 3: self.token_triggers['END']}

        if current_state in progress_bonuses:
            for tid in progress_bonuses[current_state]:
                if tid < len(candidate_logits):
                    mask[tid] *= bonus_multiplier

        return candidate_logits * mask

# =====================================================
# 🔄 INTEGRATED GENERATION
# =====================================================

def generate_with_nsg(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    nsg = NeuroSymbolicGenerator(self.tokenizer)
    grammar_validator = StoryGrammarValidator(self.tokenizer)

    for _ in range(max_new_tokens):
        logits = self(idx)[:, -1, :] / temperature

        for b in range(idx.size(0)):
            logits[b] = nsg.apply_repetition_penalty(logits[b], idx[b])
            logits[b] = grammar_validator.validate_and_score(idx[b], logits[b])
            logits[b] = nsg.enforce_narrative_diversity(idx[b], logits[b])

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits[b], descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[b][indices_to_remove] = -float('Inf')

            if top_k:
                v, _ = torch.topk(logits[b], min(top_k, logits[b].size(-1)))
                logits[b][logits[b] < v[-1]] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)

    return idx

BitNetTransformer2b.generate = generate_with_nsg

# =====================================================
# 🧩 TOKENIZER WRAPPER
# =====================================================

class SimpleTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, return_tensors=None):
        ids = self.sp.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids):
        if torch.is_tensor(ids): ids = ids.tolist()
        if len(ids) == 0:
            return ""
        if isinstance(ids[0], list): ids = ids[0]
        return self.sp.decode(ids)

# =====================================================
# 🌐 DOWNLOAD DATASET
# =====================================================

print("🌐 DOWNLOADING DATASET...")

url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
filename = os.path.join(SAVE_PATH, "tinystories.txt")

if not os.path.exists(filename):
    try:
        urllib.request.urlretrieve(url, filename)
        print("✅ Dataset downloaded!")
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        sys.exit(1)
else:
    print("✅ Dataset already exists.")

# =====================================================
# 🔤 TRAIN TOKENIZER
# =====================================================

print("🔤 TRAINING TOKENIZER...")

model_prefix = os.path.join(SAVE_PATH, "bitnet2b_en")
if not os.path.exists(f"{model_prefix}.model"):
    spm.SentencePieceTrainer.train(
        input=filename,
        model_prefix=model_prefix,
        vocab_size=4096,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0, eos_id=1, bos_id=2, unk_id=3
    )
    print("✅ Tokenizer trained!")
else:
    print("✅ Tokenizer already exists.")

tokenizer = SimpleTokenizer(f"{model_prefix}.model")

# =====================================================
# 📚 DATASET CLASS
# =====================================================

class StoryDataset(Dataset):
    def __init__(self, filename, seq_len=128):
        with open(filename, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.seq_len = seq_len
        print(f"📚 Loaded {len(self.lines):,} stories for training.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokens = tokenizer.encode(self.lines[idx])
        if len(tokens) < self.seq_len + 1:
            tokens += [0] * (self.seq_len + 1 - len(tokens))
        else:
            tokens = tokens[:self.seq_len + 1]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

# =====================================================
# 🚀 TRAINING SETUP — GPU + CHECKPOINTS + AII SAMPLING
# =====================================================

print("🧠 INITIALIZING MODEL...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

model = BitNetTransformer2b(
    num_tokens=4096,
    dim=512,
    depth=8,
    dim_head=64,
    heads=8,
    ff_mult=4
).to(device).train()

model.tokenizer = tokenizer

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

# ➤➤ PATHS FOR AUTO-SAVING & UPLOADING
AUTO_SAVE_PATH = os.path.join(SAVE_PATH, "autosave_latest.pt")
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
DAILY_CKPT_PATH = os.path.join(SAVE_PATH, f"model_daily_{TODAY_DATE}.pt")

# Check if daily checkpoint already exists
daily_checkpoint_saved_today = os.path.exists(DAILY_CKPT_PATH)
if daily_checkpoint_saved_today:
    print(f"📌 Daily checkpoint for {TODAY_DATE} already exists.")
else:
    print(f"📌 Will save daily checkpoint after first epoch.")

# ➤➤ RESUME FROM AUTO-SAVE IF EXISTS (WITH ERROR HANDLING)
start_epoch = 0
start_step = 0

if os.path.exists(AUTO_SAVE_PATH):
    print(f"🔁 ATTEMPTING TO LOAD AUTO-SAVE CHECKPOINT: {AUTO_SAVE_PATH}")
    try:
        ckpt = torch.load(AUTO_SAVE_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        start_step = ckpt.get('step_in_epoch', 0)
        print(f"✅ SUCCESS! Resuming from Epoch {start_epoch + 1}, Step {start_step}")
    except Exception as e:
        print(f"❌ FAILED TO LOAD AUTO-SAVE: {e}")
        print(f"⚠️  {AUTO_SAVE_PATH} is corrupted or incomplete. Removing it to avoid future issues.")
        os.remove(AUTO_SAVE_PATH)
        print("🗑️  Corrupted autosave deleted. Will resume from latest epoch checkpoint or start fresh.")

# If no valid autosave, try epoch checkpoints
if start_epoch == 0 and start_step == 0:
    checkpoint_files = [f for f in os.listdir(SAVE_PATH) if f.startswith("checkpoint_epoch") and f.endswith(".pt")]
    if checkpoint_files:
        latest = max(checkpoint_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
        ckpt_path = os.path.join(SAVE_PATH, latest)
        print(f"🔁 LOADING EPOCH CHECKPOINT: {latest}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"▶️ Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"❌ FAILED TO LOAD EPOCH CHECKPOINT: {e}")
            print("🔄 Starting from scratch.")
            start_epoch = 0

dataset = StoryDataset(filename)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
TOTAL_EPOCHS = 10

# =====================================================
# 🧪 AII — AI INSPECTION SAMPLER (TEXT GENERATION AFTER EVERY CHECKPOINT)
# =====================================================

def generate_sample_story(model, tokenizer, prompt="the sun rises", max_new_tokens=80, temperature=0.9, top_k=50, top_p=0.90):
    """Generate a short story sample for qualitative monitoring."""
    model.eval()  # Set to eval mode for generation
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        story = tokenizer.decode(output[0])
    model.train()  # Switch back to train mode
    return story

def log_sample_to_file(sample_text, label="sample"):
    """Optional: Save sample to timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.txt"
    filepath = os.path.join(SAVE_PATH, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(f"📝 Sample saved to: {filename}")

print(f"🚀 STARTING TRAINING FROM EPOCH {start_epoch + 1} for {TOTAL_EPOCHS} EPOCHS")
print("="*60)

# =====================================================
# 🏋️ TRAINING LOOP — WITH AII SAMPLING + SAFE SAVING
# =====================================================

AUTO_SAVE_EVERY_N_STEPS = 100  # Save and sample every 100 steps

for epoch in range(start_epoch, start_epoch + TOTAL_EPOCHS):
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{start_epoch + TOTAL_EPOCHS}")

    # Resume mid-epoch if needed
    if epoch == start_epoch and start_step > 0:
        print(f"⏩ Fast-forwarding to step {start_step} in epoch {epoch+1}...")
        dataloader_iter = iter(dataloader)
        for _ in range(start_step):
            next(dataloader_iter, None)
        data_iter = enumerate(dataloader_iter, start=start_step)
    else:
        data_iter = enumerate(dataloader)
        start_step = 0

    for i, (x, y) in data_iter:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        # ➤➤ AUTO-SAVE + AII SAMPLE (SAFE SAVE)
        if (i + 1) % AUTO_SAVE_EVERY_N_STEPS == 0:
            save_dict = {
                'epoch': epoch,
                'step_in_epoch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'resume_marker': 'autosave',
            }
            safe_save_checkpoint(save_dict, AUTO_SAVE_PATH)

            # ➤➤ AII — GENERATE SAMPLE STORY
            print("\n" + "-"*70)
            print(f"🧪 AII SAMPLE — Epoch {epoch+1}, Step {i+1}")
            print("-"*70)
            sample_story = generate_sample_story(model, tokenizer)
            print(sample_story)
            print("-"*70 + "\n")

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # ➤➤ SAVE EPOCH CHECKPOINT + AII SAMPLE (SAFE SAVE)
    ckpt_path = os.path.join(SAVE_PATH, f"checkpoint_epoch{epoch+1}.pt")
    epoch_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
    }
    safe_save_checkpoint(epoch_dict, ckpt_path)

    # Generate sample after epoch save
    print("\n" + "="*70)
    print(f"🧪 AII SAMPLE — END OF EPOCH {epoch+1}")
    print("="*70)
    sample_story = generate_sample_story(model, tokenizer, max_new_tokens=100)
    print(sample_story)
    print("="*70 + "\n")

    # ➤➤ SAVE DAILY CHECKPOINT + AII SAMPLE (once per calendar day)
    if not daily_checkpoint_saved_today:
        daily_dict = {
            'epoch': epoch,
            'date': TODAY_DATE,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'trained_epochs_today': epoch + 1 - start_epoch,
        }
        safe_save_checkpoint(daily_dict, DAILY_CKPT_PATH)
        print(f"🌞 DAILY CHECKPOINT SAVED: {os.path.basename(DAILY_CKPT_PATH)}")

        # Generate sample after daily save
        print("\n" + "🌞"*35)
        print(f"🌞 AII DAILY SAMPLE — {TODAY_DATE}")
        print("🌞"*35)
        sample_story = generate_sample_story(model, tokenizer, max_new_tokens=120, temperature=0.85)
        print(sample_story)
        print("🌞"*35 + "\n")

        daily_checkpoint_saved_today = True

print("="*60)
print("🎉 TRAINING COMPLETE!")

# =====================================================
# 📖 FINAL GENERATION — SHOW OFF YOUR MODEL
# =====================================================

print("📖 GENERATING FINAL NEURO-SYMBOLIC STORY...")

model.eval()
prompt = "In a land far away"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

output = model.generate(input_ids, max_new_tokens=150, temperature=0.95, top_k=80, top_p=0.92)
final_story = tokenizer.decode(output[0])

print("\n" + "="*70)
print("🏆 FINAL MODEL OUTPUT — AFTER TRAINING")
print("="*70)
print(final_story)
print("="*70)

# =====================================================
# 💾 SAVE FINAL MODEL + CONFIG + ZIP → GOOGLE DRIVE
# =====================================================

print("💾 SAVING FINAL ARTIFACTS TO GOOGLE DRIVE...")

final_model_path = os.path.join(SAVE_PATH, "bitnet2b_final.pt")
safe_save_checkpoint(model.state_dict(), final_model_path)

config = {
    "model": "BitNet-NSG 2b++",
    "vocab_size": 4096,
    "dim": 512,
    "depth": 8,
    "neurosymbolic": True,
    "grammar_validator": True,
    "sampling": {"temperature": 0.95, "top_k": 80, "top_p": 0.92},
    "trained_on": "TinyStories",
    "license": "MIT",
    "last_trained": TODAY_DATE,
    "total_epochs": TOTAL_EPOCHS,
    "final_loss": avg_loss
}
config_path = os.path.join(SAVE_PATH, "bitnet_nsg_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

# Create ZIP
zip_filename = f"bitnet2b_nsg_trained_{TODAY_DATE}.zip"
zip_path = os.path.join(SAVE_PATH, zip_filename)
files_to_zip = [
    f"{model_prefix}.model",
    f"{model_prefix}.vocab",
    final_model_path,
    config_path,
]

# Include autosave and daily if they exist
if os.path.exists(AUTO_SAVE_PATH):
    files_to_zip.append(AUTO_SAVE_PATH)
if os.path.exists(DAILY_CKPT_PATH):
    files_to_zip.append(DAILY_CKPT_PATH)

# Include all epoch checkpoints
for epoch in range(1, TOTAL_EPOCHS + 1):
    ckpt = os.path.join(SAVE_PATH, f"checkpoint_epoch{epoch}.pt")
    if os.path.exists(ckpt):
        files_to_zip.append(ckpt)

with zipfile.ZipFile(zip_path, "w") as zf:
    for f in files_to_zip:
        if os.path.exists(f):
            arcname = os.path.basename(f)
            zf.write(f, arcname)
            print(f"➕ Added {arcname}")
        else:
            print(f"⚠️  Skipped {f} — not found")

print(f"\n✅ FINAL ZIP SAVED TO GOOGLE DRIVE: {zip_filename}")
print(f"📁 Location: {SAVE_PATH}")
print("✅ CONGRATULATIONS — YOU TRAINED A NEURO-SYMBOLIC 2-BIT LLM WITH AI INSPECTION (AII) — FULLY AUTOMATED, CRASH-RESILIENT & CLOUD-READY!")
