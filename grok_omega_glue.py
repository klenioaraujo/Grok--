import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from tqdm import tqdm
import time

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== OPERAÇÕES ΨQRH DOE-COMPLIANT ====================

class QuaternionFFT(nn.Module):
    """Camada FFT Quaterniônica"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=1)

class QuaternionIFFT(nn.Module):
    """Camada IFFT Quaterniônica"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(x, dim=1).real

class HamiltonProduct(nn.Module):
    """Produto Hamiltoniano - SEM softmax"""
    def __init__(self):
        super().__init__()
    
    def forward(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # Garantir dimensões
        if q1.dim() == 2:
            q1 = q1.unsqueeze(0)
        if q2.dim() == 2:
            q2 = q2.unsqueeze(0)
        
        # Extrair componentes com split seguro
        dim_4 = -2
        components1 = torch.split(q1, 1, dim=dim_4)
        components2 = torch.split(q2, 1, dim=dim_4)
        
        w1 = components1[0].squeeze(dim_4)
        x1 = components1[1].squeeze(dim_4)
        y1 = components1[2].squeeze(dim_4)
        z1 = components1[3].squeeze(dim_4)
        
        w2 = components2[0].squeeze(dim_4)
        x2 = components2[1].squeeze(dim_4)
        y2 = components2[2].squeeze(dim_4)
        z2 = components2[3].squeeze(dim_4)
        
        # Produto Hamiltoniano (substitui atenção com softmax)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        result = torch.stack([w, x, y, z], dim=dim_4)
        
        if len(q1.shape) == 2:
            result = result.squeeze(0)
            
        return result

class LogarithmicSpectralFilter(nn.Module):
    """Filtro espectral logarítmico - conforme DOE"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.quat_dim = d_model // 4
        self.alpha = nn.Parameter(torch.ones(self.quat_dim) * 0.1)
        self.epsilon = 1e-8
        
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        B, T, C, D = k.shape
        magnitude = torch.abs(k) + self.epsilon
        log_mag = torch.log(magnitude)
        phase = self.alpha.view(1, 1, 1, D) * torch.atan(log_mag)
        real = torch.cos(phase)
        imag = torch.sin(phase)
        return torch.complex(real, imag)

class SpectralInterference(nn.Module):
    """Interferência Espectral - substitui atenção com softmax"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.quat_dim = d_model // 4
        
        self.fft = QuaternionFFT()
        self.ifft = QuaternionIFFT()
        self.spectral_filter = LogarithmicSpectralFilter(d_model)
        self.hamilton = HamiltonProduct()
        
        # Projeções quaterniônicas (substituem Q, K, V)
        self.Q_proj = nn.Linear(d_model, d_model)
        self.R_proj = nn.Linear(d_model, d_model)
        self.H_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Projetar para quaterniões
        Q = self.Q_proj(x).view(B, T, 4, self.quat_dim)
        R = self.R_proj(x).view(B, T, 4, self.quat_dim)
        H = self.H_proj(x).view(B, T, 4, self.quat_dim)
        
        # Domínio espectral
        Q_spectral = self.fft(Q)
        R_spectral = self.fft(R)
        H_spectral = self.fft(H)
        
        # Filtro espectral (substitui softmax)
        Q_filtered = Q_spectral * self.spectral_filter(Q_spectral)
        R_filtered = R_spectral * self.spectral_filter(R_spectral)
        H_filtered = H_spectral * self.spectral_filter(H_spectral)
        
        # Interferência espectral (substitui Q@K^T)
        QR_product = self.hamilton(Q_filtered, R_filtered)
        interference_spectral = self.hamilton(QR_product, H_filtered)  # Ψ(Q,R,H)
        
        # Voltar para temporal
        interference_temporal = self.ifft(interference_spectral)
        output = interference_temporal.reshape(B, T, -1)
        
        return self.norm(output)

class HamiltonianEvolution(nn.Module):
    """Evolução Hamiltoniana - substitui FFN tradicional"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.quat_dim = d_model // 4
        
        self.fft = QuaternionFFT()
        self.ifft = QuaternionIFFT()
        self.hamilton = HamiltonProduct()
        
        # Quaterniões unitários para rotação
        self.q_left = nn.Parameter(torch.zeros(4, self.quat_dim))
        self.q_right = nn.Parameter(torch.zeros(4, self.quat_dim))
        self.q_left.data[0] = 1.0  # Inicializar como identidade
        self.q_right.data[0] = 1.0
        
        self.spectral_gate = nn.Parameter(torch.ones(1, 1, 1, self.quat_dim))
        
    def _normalize_quaternion(self, q: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(q**2, dim=0, keepdim=True) + 1e-8)
        return q / norm
    
    def _quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        conjugate = q.clone()
        conjugate[1:, :] = -conjugate[1:, :]
        return conjugate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_quat = x.view(B, T, 4, self.quat_dim)
        
        # Processamento espectral
        x_spectral = self.fft(x_quat)
        filtered_spectral = x_spectral * self.spectral_gate
        x_filtered = self.ifft(filtered_spectral)
        
        # Rotações quaterniônicas
        q_left_norm = self._normalize_quaternion(self.q_left)
        q_right_norm = self._normalize_quaternion(self.q_right)
        q_right_conj = self._quaternion_conjugate(q_right_norm)
        
        # Aplicar rotações por posição (evita broadcasting complexo)
        x_rotated = []
        for t in range(T):
            x_t = x_filtered[:, t, :, :]
            q_left_exp = q_left_norm.unsqueeze(0)
            q_right_conj_exp = q_right_conj.unsqueeze(0)
            
            x_right = self.hamilton(x_t, q_right_conj_exp)
            x_t_rotated = self.hamilton(q_left_exp, x_right)
            x_rotated.append(x_t_rotated)
        
        x_rotated = torch.stack(x_rotated, dim=1)
        output = x_rotated.reshape(B, T, -1)
        
        return output

# ==================== MODELO ΨQRH PARA GLUE ====================

class PsiQRHForGLUE(nn.Module):
    """Modelo ΨQRH para tasks GLUE - 100% DOE-compliant"""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=6, num_labels=2):
        super().__init__()
        assert hidden_size % 4 == 0, "hidden_size deve ser divisível por 4"
        
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        
        # Embeddings (sem positional encoding tradicional)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)  # Posições fixas
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Camadas ΨQRH (substituem transformer layers)
        self.spectral_layers = nn.ModuleList([
            SpectralInterference(hidden_size) for _ in range(num_layers)
        ])
        self.hamiltonian_layers = nn.ModuleList([
            HamiltonianEvolution(hidden_size) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Classificador (sem softmax na saída)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.tanh = nn.Tanh()
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.shape
        
        # Embeddings + Posições
        word_embeddings = self.word_embeddings(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        embeddings = self.embedding_norm(embeddings)
        hidden_states = self.dropout(embeddings)
        
        # Encoder ΨQRH (sem atenção tradicional)
        for spectral, hamiltonian, norm in zip(
            self.spectral_layers, self.hamiltonian_layers, self.layer_norms
        ):
            residual = hidden_states
            hidden_states = spectral(hidden_states)  # Interferência espectral
            hidden_states = hamiltonian(hidden_states)  # Evolução Hamiltoniana
            hidden_states = norm(hidden_states + residual)
        
        # Pooling para classificação
        pooled_output = hidden_states[:, 0]  # Token [CLS]
        pooled_output = self.pooler(pooled_output)
        pooled_output = self.tanh(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        # Loss sem softmax (usamos CrossEntropy que aplica softmax internamente)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

# ==================== PIPELINE GLUE COMPLETO ====================

class GLUEConfig:
    """Configuração para tasks GLUE"""
    def __init__(self, task_name="sst2"):
        self.task_name = task_name
        self.max_length = 128
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.epochs = 3
        self.warmup_steps = 100
        self.weight_decay = 0.01
        
        task_configs = {
            "sst2": {"num_labels": 2, "metric": "accuracy"},
            "mnli": {"num_labels": 3, "metric": "accuracy"},
            "qqp": {"num_labels": 2, "metric": "accuracy"},
            "qnli": {"num_labels": 2, "metric": "accuracy"},
        }
        
        self.num_labels = task_configs[task_name]["num_labels"]
        self.metric_name = task_configs[task_name]["metric"]

class GLUEDataProcessor:
    """Processador de dados GLUE simplificado"""
    
    def __init__(self, config):
        self.config = config
        self.vocab_size = 30522
        
    def load_dataset(self, split='train'):
        """Carrega dataset simulado para demonstração"""
        if split == 'train':
            texts = [
                "excellent movie fantastic acting brilliant story",
                "terrible film awful acting boring plot",
                "amazing performance wonderful cinematography",
                "poor execution disappointing results waste of time",
                "outstanding direction superb acting",
                "bad movie terrible script poor acting",
                "masterpiece of cinema brilliant filmmaking", 
                "awful film complete disaster boring",
            ] * 50  # 400 exemplos
            
            labels = [1, 0, 1, 0, 1, 0, 1, 0] * 50
        else:
            texts = [
                "great film with excellent performances",
                "terrible movie with bad acting",
                "wonderful story and amazing acting",
                "poor quality and disappointing",
            ] * 10  # 40 exemplos
            
            labels = [1, 0, 1, 0] * 10
            
        return {'text': texts, 'label': labels}
    
    def preprocess_batch(self, examples):
        """Pré-processa batch de exemplos"""
        # Tokenização simplificada
        batch_size = len(examples['text'])
        input_ids = []
        
        for text in examples['text']:
            # Simulação de tokenização BERT
            tokens = text.lower().split()[:self.config.max_length]
            tokens_ids = [hash(token) % self.vocab_size for token in tokens]
            
            # Padding
            if len(tokens_ids) < self.config.max_length:
                tokens_ids += [0] * (self.config.max_length - len(tokens_ids))
            else:
                tokens_ids = tokens_ids[:self.config.max_length]
                
            input_ids.append(tokens_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor([[1 if x != 0 else 0 for x in seq] for seq in input_ids]),
            'labels': torch.tensor(examples['label'], dtype=torch.long)
        }

class GLUETrainer:
    """Treinador para ΨQRH em tasks GLUE"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Otimizador
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.epochs * 100
        )
        
    def train(self, train_dataset, eval_dataset=None):
        """Treina o modelo ΨQRH"""
        logger.info("🚀 Iniciando treinamento ΨQRH para GLUE")
        
        processor = GLUEDataProcessor(self.config)
        train_data = processor.load_dataset('train')
        train_processed = processor.preprocess_batch(train_data)
        
        training_stats = []
        
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            
            # Batch único para simplificação
            batch = {k: v.to(self.device) for k, v in train_processed.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss = loss.item()
            
            # Avaliação
            if eval_dataset:
                eval_results = self.evaluate(eval_dataset)
                training_stats.append({
                    'epoch': epoch + 1,
                    'train_loss': total_loss,
                    **eval_results
                })
                
                logger.info(f"📊 Época {epoch+1}: Loss={total_loss:.4f}, "
                           f"Accuracy={eval_results.get('accuracy', 0):.4f}")
            else:
                logger.info(f"📊 Época {epoch+1}: Loss={total_loss:.4f}")
        
        return training_stats
    
    def evaluate(self, eval_dataset):
        """Avalia o modelo"""
        self.model.eval()
        processor = GLUEDataProcessor(self.config)
        eval_data = processor.load_dataset('validation')
        eval_processed = processor.preprocess_batch(eval_data)
        
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in eval_processed.items()}
            outputs = self.model(**batch)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch['labels']).float().mean()
        
        return {'accuracy': accuracy.item()}
    
    def predict(self, text):
        """Faz predição em texto individual"""
        self.model.eval()
        processor = GLUEDataProcessor(self.config)
        
        inputs = processor.preprocess_batch({'text': [text], 'label': [0]})
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
        
        return {
            'prediction': prediction.cpu().item(),
            'probabilities': probabilities.cpu().numpy()[0]
        }

# ==================== PIPELINE PRINCIPAL ====================

def main():
    """Pipeline completo GLUE com ΨQRH"""
    print("=" * 70)
    print("🎯 GLUE PIPELINE COM ΨQRH (DOE-COMPLIANT)")
    print("=" * 70)
    print("✅ SEM softmax attention")
    print("✅ SEM transformer tradicional") 
    print("✅ 100% DOE-compliant")
    print("=" * 70)
    
    # Configuração
    config = GLUEConfig("sst2")
    
    # Modelo ΨQRH
    model = PsiQRHForGLUE(
        vocab_size=30522,
        hidden_size=768,
        num_layers=6,
        num_labels=config.num_labels
    )
    
    # Treinador
    trainer = GLUETrainer(model, config)
    
    # Treinamento
    print("\n🚀 Iniciando treinamento...")
    start_time = time.time()
    stats = trainer.train(None, None)  # Usa datasets internos
    training_time = time.time() - start_time
    
    print(f"\n✅ Treinamento concluído em {training_time:.2f}s")
    
    # Predições de teste
    print("\n🔍 TESTE DE PREDIÇÕES:")
    print("-" * 50)
    
    test_cases = [
        "This movie is absolutely fantastic and wonderful",
        "Terrible film with awful acting and boring story",
        "Amazing cinematography and brilliant performances",
        "Poor execution and disappointing results"
    ]
    
    for text in test_cases:
        result = trainer.predict(text)
        sentiment = "POSITIVE" if result['prediction'] == 1 else "NEGATIVE"
        confidence = result['probabilities'][result['prediction']]
        
        print(f"📝 '{text}'")
        print(f"   → {sentiment} (confiança: {confidence:.4f})")
        print(f"   → Probs: [NEG: {result['probabilities'][0]:.4f}, "
              f"POS: {result['probabilities'][1]:.4f}]")
        print()

if __name__ == "__main__":
    main()