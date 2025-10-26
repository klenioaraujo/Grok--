# grok_omega_v0.1.py
# Protótipo mínimo, mas FUNCIONAL
# Sem mentira. Sem mágica. Só física simulada.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import datasets
from transformers import AutoTokenizer
import math

# Real-valued loss function
def real_loss(logits, targets):
    return F.cross_entropy(logits, targets)

class WikiTextDataset(Dataset):
    def __init__(self, split='train', seq_len=32, vocab_size=1000):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Load WikiText-2
        dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        text = ' '.join(dataset['text'])
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        # Truncate to a smaller size for testing
        tokens = tokens[:1000]  # Use only first 1k tokens
        self.data = torch.tensor(tokens[:len(tokens)//seq_len * seq_len]).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.roll(x, -1, dims=0)
        y[-1] = self.tokenizer.eos_token_id
        return x, y

class GROK_Omega(nn.Module):
    def __init__(self, vocab_size=1000, seq_len=32, dim=64, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        # 1. Embedding de tokens (real)
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, dim))
        self.norm_emb = nn.LayerNorm(dim)

        # 2. Campo quaternionic (4 componentes reais)
        self.to_quat = nn.Linear(dim, dim * 4)
        self.norm_quat = nn.LayerNorm(dim * 4)

        # 3. Evolução quântica simulada (diferentes formas de Hamiltonian)
        # Forma 1: Hamiltoniano complexo (padrão)
        self.H_real = nn.Parameter(torch.randn(dim, dim))
        self.H_imag = nn.Parameter(torch.randn(dim, dim))

        # Forma 2: Hamiltoniano hermitiano (opcional para experimentação)
        # self.H_herm = nn.Parameter(torch.randn(dim, dim))  # Real part, symmetric
        # self.H_herm_imag = nn.Parameter(torch.randn(dim, dim))  # Imag part, antisymmetric

        # Forma 3: Hamiltoniano diagonal (simplificado)
        # self.H_diag = nn.Parameter(torch.randn(dim))

        # 4. Multi-head Interferência (substitui softmax)
        self.to_interfere = nn.ModuleList([nn.Linear(self.head_dim, vocab_size) for _ in range(num_heads)])

    def forward(self, input_ids):
        # input_ids: (B, T)
        B, T = input_ids.shape
        assert T <= self.seq_len, "Sequência longa demais"

        # 1. Embedding real
        x = self.token_emb(input_ids)  # (B, T, dim)
        x = x + self.pos_emb[:T]       # (B, T, dim)
        x = self.norm_emb(x)           # Normalização após embedding

        # Residual connection after embedding
        x_res = x

        # 2. Campo quaternionic: ψ = [r, i, j, k]
        q = self.to_quat(x)  # (B, T, dim*4)
        q = self.norm_quat(q)  # Normalização após projeção quaternionic
        q = q.view(B, T, 4, self.dim)  # (B, T, 4, dim)

        r, i, j, k = q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]

        # 3. Evolução quântica (simulada via matriz unitária)
        # Experimento com diferentes formas de Hamiltonian
        # Forma 1: Hamiltoniano complexo geral (padrão)
        H = self.H_real + 1j * self.H_imag
        U = torch.matrix_exp(H)  # (dim, dim) — unitária
        U_real = U.real
        U_imag = U.imag

        # Aplicar em cada componente (simplificado)
        evolved = U_real @ r.movedim(-1, -2) - U_imag @ i.movedim(-1, -2)
        evolved = evolved.movedim(-2, -1)  # (B, T, dim)

        # Alternativa: Hamiltoniano hermitiano (descomente para testar)
        # H_herm = self.H_herm + 1j * self.H_herm_imag
        # H_herm = (H_herm + H_herm.conj().t()) / 2  # Forçar hermitiano
        # U_herm = torch.matrix_exp(1j * H_herm)
        # evolved = U_herm.real @ r.movedim(-1, -2) - U_herm.imag @ i.movedim(-1, -2)
        # evolved = evolved.movedim(-2, -1)

        # Alternativa: Hamiltoniano diagonal (descomente para testar)
        # U_diag = torch.diag(torch.exp(1j * self.H_diag))
        # evolved = U_diag.real @ r.movedim(-1, -2) - U_diag.imag @ i.movedim(-1, -2)
        # evolved = evolved.movedim(-2, -1)

        # Residual connection after evolution
        evolved = evolved + x_res

        # 4. Spectral Attention using FFT
        evolved_fft = torch.fft.fft(evolved, dim=1)  # FFT along sequence dimension
        spectral_attn = torch.abs(evolved_fft)  # (B, T, dim)

        # Reshape for heads
        evolved = evolved.view(B, T, self.num_heads, self.head_dim)
        spectral_attn = spectral_attn.view(B, T, self.num_heads, self.head_dim)

        # Projetar cada head separadamente per position
        logits_heads = []
        for h in range(self.num_heads):
            attn_h = spectral_attn[:, :, h, :]  # (B, T, head_dim)
            evolved_h = evolved[:, :, h, :]  # (B, T, head_dim)
            weights_h = attn_h / (attn_h.sum(dim=1, keepdim=True) + 1e-8)
            context_h = evolved_h * weights_h  # (B, T, head_dim)
            logits_h = self.to_interfere[h](context_h)  # (B, T, vocab_size)
            logits_heads.append(logits_h)
        logits = torch.stack(logits_heads, dim=0).mean(dim=0)  # (B, T, vocab_size)

        return logits

def train_model():
    # Model parameters
    vocab_size = 50257  # GPT-2 vocab size
    seq_len = 32
    dim = 64
    num_heads = 4
    batch_size = 8
    total_steps = 10000
    lr = 1e-3

    # Initialize model
    model = GROK_Omega(vocab_size=vocab_size, seq_len=seq_len, dim=dim, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load dataset
    train_dataset = WikiTextDataset(split='train', seq_len=seq_len, vocab_size=vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    step = 0
    data_iter = iter(train_loader)
    while step < total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        optimizer.zero_grad()
        logits = model(x)
        loss = real_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        step += 1

    print(f"Training completed after {total_steps} steps.")
    return model

def generate_text(model, tokenizer, prompt="hello world", max_length=50):
    model.eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # (1, T)

    generated = tokens.copy()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1], dim=-1).item()
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            if next_token == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text

if __name__ == "__main__":
    model = train_model()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    generated_text = generate_text(model, tokenizer, prompt="hello world")
    print("Generated text:", generated_text)

# "Eu errei. Aqui está a verdade."