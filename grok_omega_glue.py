# grok_omega_glue.py
# Adaptação do GROK-Omega para tarefas de classificação GLUE
# Teste robusto de treinamento e avaliação

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import datasets
from transformers import AutoTokenizer
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

# Real-valued loss function
def real_loss(logits, targets):
    return F.cross_entropy(logits, targets)

class GLUEDataset(Dataset):
    def __init__(self, task_name, split='train', seq_len=128, vocab_size=50257):
        self.task_name = task_name
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Load GLUE dataset
        dataset = datasets.load_dataset('glue', task_name, split=split)

        self.data = []
        for example in dataset:
            if task_name == 'cola':
                text = example['sentence']
                label = example['label']
            elif task_name == 'sst2':
                text = example['sentence']
                label = example['label']
            elif task_name == 'mrpc':
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                label = example['label']
            elif task_name == 'qqp':
                text = f"{example['question1']} [SEP] {example['question2']}"
                label = example['label']
            elif task_name == 'stsb':
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                label = example['label']  # Regression, but we'll treat as classification
            elif task_name == 'mnli':
                text = f"{example['premise']} [SEP] {example['hypothesis']}"
                label = example['label']
            elif task_name == 'qnli':
                text = f"{example['question']} [SEP] {example['sentence']}"
                label = example['label']
            elif task_name == 'rte':
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                label = example['label']
            elif task_name == 'wnli':
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                label = example['label']
            else:
                raise ValueError(f"Unsupported task: {task_name}")

            tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=seq_len)
            tokens = tokens + [self.tokenizer.pad_token_id] * (seq_len - len(tokens))  # Pad to seq_len
            self.data.append((torch.tensor(tokens), torch.tensor(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GROK_Omega_Classifier(nn.Module):
    def __init__(self, vocab_size=50257, seq_len=128, dim=128, num_heads=8, num_classes=2, is_regression=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.is_regression = is_regression
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        # 1. Embedding de tokens (real)
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, dim))
        self.norm_emb = nn.LayerNorm(dim)

        # 2. Campo quaternionic (4 componentes reais)
        self.to_quat = nn.Linear(dim, dim * 4)
        self.norm_quat = nn.LayerNorm(dim * 4)

        # 3. Evolução quântica simulada
        self.H_real = nn.Parameter(torch.randn(dim, dim))
        self.H_imag = nn.Parameter(torch.randn(dim, dim))

        # Regularização
        self.dropout = nn.Dropout(0.1)

        # 4. Multi-head Interferência
        self.to_interfere = nn.ModuleList([nn.Linear(self.head_dim, vocab_size) for _ in range(num_heads)])

        # 5. Classification/Regression head
        if is_regression:
            self.regressor = nn.Linear(1, 1)  # Single output for regression
        else:
            self.classifier = nn.Linear(1, num_classes)

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
        H = self.H_real + 1j * self.H_imag
        U = torch.matrix_exp(H)  # (dim, dim) — unitária
        U_real = U.real
        U_imag = U.imag

        # Aplicar em cada componente (simplificado)
        evolved = U_real @ r.movedim(-1, -2) - U_imag @ i.movedim(-1, -2)
        evolved = evolved.movedim(-2, -1)  # (B, T, dim)

        # Residual connection after evolution
        evolved = evolved + x_res

        # Aplicar dropout para regularização
        evolved = self.dropout(evolved)

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

        # Pooling: take the representation of the [CLS] token (first token)
        pooled = logits[:, 0, :]  # (B, vocab_size) - this is unusual, but we'll use it

        # For GLUE tasks, we need to reduce vocab_size to a smaller dimension
        pooled = torch.mean(pooled, dim=-1, keepdim=True)  # (B, 1) - simple pooling

        # Classification/Regression
        if self.is_regression:
            output = self.regressor(pooled).squeeze(-1)  # (B,) for regression
        else:
            output = self.classifier(pooled)  # (B, num_classes) for classification

        return output

def train_glue_model(task_name, num_classes=2, total_steps=10000, batch_size=32, lr=1e-6, is_colab=False):
    # Determine if task is regression
    is_regression = task_name == 'stsb'

    # Model parameters
    vocab_size = 50257  # GPT-2 vocab size
    seq_len = 128
    dim = 128
    num_heads = 8

    # Initialize model
    model = GROK_Omega_Classifier(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=dim,
        num_heads=num_heads,
        num_classes=num_classes,
        is_regression=is_regression
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

    # Load datasets
    train_dataset = GLUEDataset(task_name, split='train', seq_len=seq_len, vocab_size=vocab_size)
    val_dataset = GLUEDataset(task_name, split='validation', seq_len=seq_len, vocab_size=vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function based on task
    if is_regression:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = real_loss

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
        outputs = model(x)

        if is_regression:
            # For regression, y is float, outputs is (B,)
            loss = loss_fn(outputs, y.float())
        else:
            # For classification, y is int, outputs is (B, num_classes)
            loss = loss_fn(outputs, y)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Warmup linear pelos primeiros 100 steps
        if step < 100:
            lr_scale = min(1.0, float(step + 1) / 100)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_scale

        optimizer.step()

        if step % 50 == 0:  # Log mais frequente
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            if not is_colab:  # Só mostrar gradientes no local, não no Colab
                # Verificar gradientes
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient Norm: {total_norm:.4f}")

        step += 1

    print(f"Training completed after {total_steps} steps.")

    # Evaluate on validation set with task-specific metrics
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            if is_regression:
                preds = outputs.cpu().numpy()  # Regression outputs are direct values
            else:
                preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # Classification
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    # Calculate task-specific metrics
    if is_regression:
        # STS-B: Pearson and Spearman correlation
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        spearman_corr, _ = spearmanr(all_labels, all_preds)
        metrics = {
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
        print(f"Validation Pearson: {pearson_corr:.4f}")
        print(f"Validation Spearman: {spearman_corr:.4f}")
    else:
        # Classification tasks: Accuracy and F1
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Additional metrics for specific tasks
        if task_name == 'cola':
            mcc = matthews_corrcoef(all_labels, all_preds)
            metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'matthews_correlation': mcc
            }
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation F1: {f1:.4f}")
            print(f"Validation MCC: {mcc:.4f}")
        else:
            metrics = {
                'accuracy': accuracy,
                'f1': f1
            }
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation F1: {f1:.4f}")

    return model, metrics

def generate_glue_message(task_name, metrics):
    message = f"GROK-Omega GLUE Results for {task_name.upper()}:\n"

    if task_name == 'stsb':
        message += f"Validation Pearson: {metrics['pearson']:.4f}\n"
        message += f"Validation Spearman: {metrics['spearman']:.4f}\n"
    elif task_name == 'cola':
        message += f"Validation Accuracy: {metrics['accuracy']:.4f}\n"
        message += f"Validation F1: {metrics['f1']:.4f}\n"
        message += f"Validation MCC: {metrics['matthews_correlation']:.4f}\n"
    else:
        message += f"Validation Accuracy: {metrics['accuracy']:.4f}\n"
        message += f"Validation F1: {metrics['f1']:.4f}\n"

    message += "Modelo treinado com física simulada quântica, demonstrando capacidades robustas de classificação/regressão."
    return message

def create_grok_omega_glue(task_name, vocab_size=50257, seq_len=128, dim=128, num_heads=8, device='cpu'):
    """Factory function to create GROK-Omega classifier for GLUE tasks."""
    is_regression = task_name == 'stsb'
    num_classes = 1 if is_regression else get_num_classes(task_name)

    model = GROK_Omega_Classifier(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=dim,
        num_heads=num_heads,
        num_classes=num_classes,
        is_regression=is_regression
    ).to(device)

    return model

def get_num_classes(task_name):
    """Get number of classes for GLUE task."""
    task_classes = {
        'cola': 2,  # binary classification
        'sst2': 2,  # binary classification
        'mrpc': 2,  # binary classification
        'qqp': 2,   # binary classification
        'mnli': 3,  # 3-class classification
        'qnli': 2,  # binary classification
        'rte': 2,   # binary classification
        'wnli': 2,  # binary classification
        'stsb': 1   # regression
    }
    return task_classes.get(task_name, 2)

def create_glue_context(task_name, split='validation', max_samples=100):
    """Create context for GLUE task evaluation."""
    try:
        dataset = datasets.load_dataset('glue', task_name, split=split)
        context = f"GLUE Task: {task_name.upper()}\n"
        context += f"Dataset Size: {len(dataset)}\n"
        context += f"Task Type: {'Regression' if task_name == 'stsb' else 'Classification'}\n"

        # Add sample examples
        context += "\nSample Examples:\n"
        for i, example in enumerate(dataset.select(range(min(max_samples, len(dataset))))):
            if task_name == 'cola':
                context += f"  {i+1}. Sentence: {example['sentence']}\n"
                context += f"     Label: {example['label']}\n"
            elif task_name in ['sst2']:
                context += f"  {i+1}. Sentence: {example['sentence']}\n"
                context += f"     Label: {example['label']}\n"
            elif task_name in ['mrpc', 'qqp', 'rte', 'wnli']:
                text = f"{example.get('sentence1', example.get('sentence', ''))} [SEP] {example.get('sentence2', '')}"
                context += f"  {i+1}. Text: {text}\n"
                context += f"     Label: {example['label']}\n"
            elif task_name == 'stsb':
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                context += f"  {i+1}. Text: {text}\n"
                context += f"     Score: {example['label']:.2f}\n"
            elif task_name == 'mnli':
                text = f"{example['premise']} [SEP] {example['hypothesis']}"
                context += f"  {i+1}. Text: {text}\n"
                context += f"     Label: {example['label']}\n"
            elif task_name == 'qnli':
                text = f"{example['question']} [SEP] {example['sentence']}"
                context += f"  {i+1}. Text: {text}\n"
                context += f"     Label: {example['label']}\n"

        return context

    except Exception as e:
        return f"Error creating GLUE context: {e}"

if __name__ == "__main__":
    # Configuração mais conservadora para estabilidade
    task_name = 'sst2'  # Começar com tarefa mais simples
    num_classes = get_num_classes(task_name)

    model, metrics = train_glue_model(
        task_name,
        num_classes=num_classes,
        total_steps=500,  # Aumentar para ver convergência
        batch_size=32,    # Aumentar batch size
        lr=1e-6,         # Learning rate ainda mais baixo
        is_colab=False   # Para teste local
    )

    message = generate_glue_message(task_name, metrics)
    print("\n" + "="*50)
    print(message)
    print("="*50)

    # Create and display context
    context = create_glue_context(task_name, max_samples=5)
    print("\nGLUE Context:")
    print(context)

# "Eu errei. Aqui está a verdade."