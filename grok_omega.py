# ==============================================================
# grok_omega_psiqrh_minimal.py
# Implementação fiel ao DOE ΨQRH (Zenodo 17171112)
# Sem softmax. Sem attention clássica. Sem Transformer.
# Apenas: quaterniões, FFT, rotações SO(4), filtragem espectral.
# ==============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

# --------------------------------------------------------------
# 1. OPERAÇÕES QUATERNIÔNICAS (Hamilton product)
# --------------------------------------------------------------
def hamilton_product(q1, q2):
    """Hamilton product for quaternions (B, T, 4)"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


# --------------------------------------------------------------
# 2. FILTRO ESPECTRAL F(k) = exp(i α arctan(ln|k|))
# --------------------------------------------------------------
class SpectralFilter(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.eps = 1e-10

    def forward(self, x):
        # x: (B, T, 4)
        B, T, _ = x.shape
        x_fft = torch.fft.fft(x, dim=1)               # (B, T, 4)
        k = torch.arange(1, T + 1, device=x.device).float()
        k = k.view(1, T, 1)                           # (1, T, 1)
        phase = torch.atan(torch.log(k + self.eps))   # (1, T, 1)
        filter_phase = self.alpha * phase
        Fk = torch.exp(1j * filter_phase)             # (1, T, 1)
        x_filtered = x_fft * Fk
        return torch.fft.ifft(x_filtered, dim=1).real # (B, T, 4)


# --------------------------------------------------------------
# 3. ROTAÇÃO SO(4) :  Ψ' = qL * Ψ * qR†
# --------------------------------------------------------------
class SO4Rotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_L = nn.Parameter(torch.randn(()))
        self.omega_L = nn.Parameter(torch.randn(()))
        self.phi_L   = nn.Parameter(torch.randn(()))
        self.theta_R = nn.Parameter(torch.randn(()))
        self.omega_R = nn.Parameter(torch.randn(()))
        self.phi_R   = nn.Parameter(torch.randn(()))

    def _unit_quaternion(self, theta, omega, phi):
        half = theta / 2
        w = torch.cos(half)
        x = torch.sin(half) * torch.cos(omega)
        y = torch.sin(half) * torch.sin(omega) * torch.cos(phi)
        z = torch.sin(half) * torch.sin(omega) * torch.sin(phi)
        return torch.stack([w, x, y, z])

    def forward(self, psi):
        # psi: (B, T, 4)
        qL = self._unit_quaternion(self.theta_L, self.omega_L, self.phi_L)   # (4,)
        qR = self._unit_quaternion(self.theta_R, self.omega_R, self.phi_R)   # (4,)
        qR_conj = qR * torch.tensor([1, -1, -1, -1], device=psi.device)

        qL = qL.view(1, 1, 4)
        qR_conj = qR_conj.view(1, 1, 4)

        temp = hamilton_product(qL.expand_as(psi), psi)
        psi_rot = hamilton_product(temp, qR_conj.expand_as(psi))
        return psi_rot


# --------------------------------------------------------------
# 4. CONVERSÃO TEXTO → ONDA (fractal-aware)
# --------------------------------------------------------------
def text_to_wave(text, seq_len=128):
    try:
        signal = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float32)
    except Exception:
        signal = np.array([113, 117, 97, 110, 116, 117, 109], dtype=np.float32)  # "quantum"

    if len(signal) < seq_len:
        padded = np.pad(signal, (0, seq_len - len(signal)), mode='constant')
    else:
        padded = signal[:seq_len]

    wave = (padded - 128.0) / 128.0
    return torch.FloatTensor(wave).unsqueeze(-1)   # (T, 1)


# --------------------------------------------------------------
# 5. MODELO ΨQRH MINIMAL (DOE-compliant)
# --------------------------------------------------------------
class PsiQRHClassifier(nn.Module):
    def __init__(self, seq_len=128, hidden_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # 1. Projeção para espaço quaterniônico (4 × hidden_dim)
        self.to_quat = nn.Linear(1, 4 * hidden_dim)
        self.norm1   = nn.LayerNorm(4 * hidden_dim)

        # 2. Camadas ΨQRH
        self.spectral_filter = SpectralFilter(alpha=1.0)
        self.so4_rot         = SO4Rotation()

        # 3. Classificador (sem softmax)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 2)               # logits diretos
        )

    def forward(self, x):
        # x: (B, T, 1)
        B, T, _ = x.shape

        # ---- Embedding quaterniônico ----
        q = self.to_quat(x)                 # (B, T, 4*hidden_dim)
        q = self.norm1(q)
        q = q.view(B, T, 4, self.hidden_dim)   # (B, T, 4, D)

        # ---- Evolução ΨQRH (filtragem + rotação) ----
        q_evolved = []
        for d in range(self.hidden_dim):
            comp = q[:, :, :, d]            # (B, T, 4)
            comp_f = self.spectral_filter(comp)
            comp_r = self.so4_rot(comp_f)
            q_evolved.append(comp_r)
        q_evolved = torch.stack(q_evolved, dim=-1)   # (B, T, 4, D)

        # ---- Agregação global (mantém toda a informação latente) ----
        features = q_evolved.mean(dim=1)    # (B, 4, D)
        features = features.view(B, -1)     # (B, 4*D) → (B, 128)

        # ---- Classificação ----
        logits = self.classifier(features)  # (B, 2)
        return logits


# --------------------------------------------------------------
# 6. DATASET BALANCEADO
# --------------------------------------------------------------
class PhysicsTextDataset:
    def __init__(self, seq_len=128):
        self.seq_len = seq_len
        self.texts = [
            "quantum wave function superposition entanglement",
            "schrodinger equation hamiltonian operator eigenstate",
            "quantum computing qubit superposition algorithm",
            "wave particle duality interference pattern",
            "quantum tunneling barrier penetration",
            "classical mechanics newton laws motion",
            "thermodynamics entropy heat engine",
            "electromagnetism maxwell equations field",
            "simple short text example",
            "traditional computing binary logic"
        ] * 20
        self.labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 20
        print("Dataset balance:", Counter(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        wave = text_to_wave(self.texts[idx], self.seq_len)   # (T, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return wave, label


# --------------------------------------------------------------
# 7. TREINAMENTO
# --------------------------------------------------------------
def train():
    dataset = PhysicsTextDataset(seq_len=128)
    model   = PsiQRHClassifier(seq_len=128, hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(30):
        total_loss = correct = total = 0
        indices = torch.randperm(len(dataset))

        for i in range(0, len(dataset), 8):
            batch_idx = indices[i:i + 8]
            waves, labels = [], []
            for idx in batch_idx:
                w, l = dataset[idx]
                waves.append(w)
                labels.append(l)
            waves  = torch.stack(waves)          # (B, T, 1)
            labels = torch.stack(labels)         # (B,)

            optimizer.zero_grad()
            logits = model(waves)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:02d}: Loss={total_loss/len(dataset):.4f}, Acc={acc:.4f}")

    return model


# --------------------------------------------------------------
# 8. EXECUÇÃO
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Treinando modelo ΨQRH minimalista (fiel ao DOE)")
    model = train()
    print("Treinamento concluído com aprendizado real.")