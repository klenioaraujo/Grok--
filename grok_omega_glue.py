# grok_omega_fisico_otimizado.py
# VERSÃO OTIMIZADA - FÍSICA COM MELHOR APRENDIZADO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedWavePhysics(nn.Module):
    """Embedding físico otimizado com não-linearidades controladas"""
    def __init__(self, freq_dim=128, spatial_dim=64):
        super().__init__()
        self.freq_dim = freq_dim
        
        # Espectro de frequências com inicialização física
        self.frequency_spectrum = nn.Parameter(torch.linspace(1.0, 20.0, freq_dim))
        self.phase_modulator = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, freq_dim * 2)
        )
        self.wave_coupling = nn.Linear(freq_dim, spatial_dim)
        
        # Controle de amplitude física
        self.amplitude_control = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        B, T, _ = x.shape
        
        # Modulação mais expressiva
        modulation = self.phase_modulator(x)
        amplitude, phase = modulation.chunk(2, dim=-1)
        
        # Controle físico da amplitude
        amplitude = torch.tanh(amplitude) * self.amplitude_control
        
        time = torch.linspace(0, 2*np.pi, T).to(x.device)
        time = time.unsqueeze(0).unsqueeze(-1)
        
        freqs = self.frequency_spectrum.unsqueeze(0).unsqueeze(1)
        wave_components = amplitude * torch.sin(freqs * time + phase)
        
        wave_field = self.wave_coupling(wave_components)
        return F.tanh(wave_field)

class OptimizedHamiltonian(nn.Module):
    """Hamiltoniano otimizado para melhor aprendizado"""
    def __init__(self, dim, num_components=4):
        super().__init__()
        self.dim = dim
        self.num_components = num_components
        
        # Múltiplos Hamiltonianos para diferentes componentes
        self.hamiltonians = nn.ParameterList([
            nn.Parameter(torch.zeros(dim, dim)) for _ in range(num_components)
        ])
        
        self._initialize_hamiltonians()
        
    def _initialize_hamiltonians(self):
        for hamiltonian in self.hamiltonians:
            with torch.no_grad():
                # Inicialização física: matrizes anti-simétricas
                A = torch.randn(self.dim, self.dim) * 0.05
                hamiltonian.data = (A - A.t()) / 2
                
    def get_evolution_operator(self, hamiltonian_idx, dt=0.1):
        """Operador de evolução para Hamiltoniano específico"""
        H = 1j * self.hamiltonians[hamiltonian_idx]
        
        # Exponencial exata para dimensões moderadas
        if self.dim <= 128:
            U = torch.matrix_exp(-1j * H * dt)
        else:
            # Aproximação de Padé
            I = torch.eye(self.dim, device=H.device, dtype=torch.complex64)
            U = I - 1j * H * dt - 0.5 * (H @ H) * (dt ** 2)
        
        return U
    
    def forward(self, psi, dt=0.1):
        # ψ: (B, T, 4, dim)
        B, T, C, D = psi.shape
        
        psi_evolved = []
        for comp in range(C):
            psi_comp = psi[:, :, comp, :]
            
            # Evolução com Hamiltoniano específico do componente
            U = self.get_evolution_operator(comp, dt)
            
            psi_complex = torch.view_as_complex(
                torch.stack([psi_comp, torch.zeros_like(psi_comp)], dim=-1)
            )
            psi_evolved_complex = torch.matmul(psi_complex, U.T)
            psi_evolved.append(psi_evolved_complex.real)
        
        return torch.stack(psi_evolved, dim=2)

class EnhancedInterference(nn.Module):
    """Interferência quântica com mecanismo de atenção física"""
    def __init__(self, output_dim, hidden_dim=32):
        super().__init__()
        
        # Mecanismo de interferência não-linear
        self.interference_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Controle de interferência
        self.interference_strength = nn.Parameter(torch.ones(1))
        
    def forward(self, psi_evolved):
        # ψ_evolved: (B, T, 4, dim)
        B, T, C, D = psi_evolved.shape
        
        # Análise espectral mais sofisticada
        spectral_power = torch.fft.fft(psi_evolved, dim=1)
        spectral_magnitude = torch.abs(spectral_power)
        
        # Média ponderada por frequência
        freq_weights = torch.softmax(spectral_magnitude.mean(dim=-1), dim=1)
        weighted_spectrum = torch.sum(spectral_magnitude * freq_weights.unsqueeze(-1), dim=1)
        
        # Interferência não-linear controlada
        spatial_avg = torch.mean(weighted_spectrum, dim=-1)  # (B, 4)
        interference = self.interference_net(spatial_avg)  # (B, output_dim)
        
        return interference * self.interference_strength

class OptimizedPhysicalGROK(nn.Module):
    """GROK físico otimizado para melhor aprendizado"""
    def __init__(self, output_dim=2, seq_len=256, hidden_dim=64):
        super().__init__()
        self.output_dim = output_dim
        
        self.wave_embedding = EnhancedWavePhysics(spatial_dim=hidden_dim)
        self.quantum_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4)
        )
        self.hamiltonian = OptimizedHamiltonian(hidden_dim)
        self.interference = EnhancedInterference(output_dim)
        
        # Normalização física
        self.pre_hamiltonian_norm = nn.LayerNorm(hidden_dim * 4)
        
    def forward(self, x):
        # x: (B, T, 1)
        
        # 1. Embedding físico melhorado
        wave_encoded = self.wave_embedding(x)  # (B, T, hidden_dim)
        
        # 2. Codificação quântica com não-linearidade
        quantum_space = self.quantum_encoder(wave_encoded)  # (B, T, hidden_dim * 4)
        quantum_space = self.pre_hamiltonian_norm(quantum_space)
        quantum_field = quantum_space.view(quantum_space.size(0), quantum_space.size(1), 4, -1)
        
        # 3. Evolução Hamiltoniana
        evolved_field = self.hamiltonian(quantum_field)  # (B, T, 4, hidden_dim)
        
        # 4. Interferência otimizada
        output = self.interference(evolved_field)  # (B, output_dim)
        
        # SAÍDA FÍSICA DIRETA - ZERO SOFTMAX
        return output

class BalancedPhysicsDataset:
    """Dataset com balanceamento de classes"""
    def __init__(self, texts, labels=None, seq_len=256):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.seq_len = seq_len
        
        # Balanceamento
        self._balance_dataset()
        
    def _balance_dataset(self):
        """Balanceia as classes para melhor aprendizado"""
        class_0 = [i for i, label in enumerate(self.labels) if label == 0]
        class_1 = [i for i, label in enumerate(self.labels) if label == 1]
        
        min_class_size = min(len(class_0), len(class_1))
        
        # Subamostrar a classe maior
        if len(class_0) > min_class_size:
            np.random.shuffle(class_0)
            class_0 = class_0[:min_class_size]
        if len(class_1) > min_class_size:
            np.random.shuffle(class_1)
            class_1 = class_1[:min_class_size]
        
        balanced_indices = class_0 + class_1
        np.random.shuffle(balanced_indices)
        
        self.texts = [self.texts[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]
        
    def text_to_physical_wave(self, text):
        """Transformação com features mais ricas"""
        try:
            bytes_data = text.encode('utf-8')
            byte_values = np.frombuffer(bytes_data, dtype=np.uint8)
        except:
            byte_values = np.frombuffer(b"quantum", dtype=np.uint8)
        
        if len(byte_values) < self.seq_len:
            padded = np.zeros(self.seq_len, dtype=np.float32)
            valid_len = min(len(byte_values), self.seq_len)
            padded[:valid_len] = byte_values[:valid_len]
            # Pad com padrão de decaimento
            if valid_len > 0:
                decay = np.linspace(byte_values[-1], 0, self.seq_len - valid_len)
                padded[valid_len:] = decay
        else:
            padded = byte_values[:self.seq_len].astype(np.float32)
        
        # Normalização com preservação de informação
        wave = (padded - 128.0) / 128.0
        
        # Adicionar ruído físico de baixa amplitude
        noise = np.random.normal(0, 0.01, wave.shape)
        wave += noise.astype(np.float32)
        
        return torch.FloatTensor(wave).unsqueeze(-1)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        wave = self.text_to_physical_wave(text)
        return wave, label

def optimized_training():
    """Treinamento otimizado com física melhorada"""
    print("🚀 GROK-Ω FÍSICO OTIMIZADO")
    print("=" * 60)
    
    # Dataset mais balanceado e rico
    texts = [
        "quantum wave function superposition entanglement physics",
        "classical mechanics deterministic universe newton",
        "schrodinger equation quantum state evolution wave", 
        "particle wave duality interference pattern light",
        "hamiltonian operator energy eigenvalues quantum",
        "quantum computing qubits superposition algorithms",
        "short simple text example for comparison",
        "traditional computing bits binary logic classical",
        "quantum entanglement bell inequality correlation",
        "newton laws motion gravity classical physics",
        "wave function collapse measurement problem quantum",
        "simple short example text without physics",
        "quantum tunneling barrier penetration effect",
        "classical electromagnetism maxwell equations",
        "quantum field theory particles fields",
        "thermodynamics entropy heat classical"
    ] * 15
    
    labels = [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 15
    
    dataset = BalancedPhysicsDataset(texts, labels, seq_len=192)
    
    print(f"📊 Dataset balanceado: {len(dataset)} exemplos")
    print(f"📊 Classe 0: {sum(1 for l in dataset.labels if l == 0)}")
    print(f"📊 Classe 1: {sum(1 for l in dataset.labels if l == 1)}")
    
    model = OptimizedPhysicalGROK(output_dim=2, hidden_dim=48, seq_len=192)
    
    # Otimização mais agressiva
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    
    epochs = 12
    batch_size = 16
    
    print(f"\n🎯 Configuração de treinamento:")
    print(f"   Épocas: {epochs}, Batch: {batch_size}, LR: {2e-4}")
    print("=" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        indices = torch.randperm(len(dataset))
        
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = []
            batch_labels = []
            
            for idx in batch_indices:
                wave, label = dataset[idx]
                batch_data.append(wave)
                batch_labels.append(label)
            
            batch_data = torch.stack(batch_data)
            batch_labels = torch.tensor(batch_labels)
            
            outputs = model(batch_data)
            loss = F.cross_entropy(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip de gradiente mais agressivo
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)
        
        scheduler.step()
        
        accuracy = correct / total
        avg_loss = total_loss / max(1, (len(dataset) // batch_size))
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Época {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | LR: {current_lr:.1e}")
        
        # Verificação física a cada 3 épocas
        if epoch % 3 == 0:
            verify_enhanced_physics(model)
    
    return model, dataset

def verify_enhanced_physics(model):
    """Verificação física otimizada"""
    print("  🔍 Física avançada:")
    
    try:
        with torch.no_grad():
            # Verificar múltiplos Hamiltonianos
            for i, hamiltonian in enumerate(model.hamiltonian.hamiltonians):
                H = 1j * hamiltonian
                hermiticity_error = (H - H.conj().t()).norm().item()
                print(f"    H{i}: {hermiticity_error:.6f}", end="")
            
            print()  # Nova linha
            
            # Verificar controle de amplitude
            amp_control = model.wave_embedding.amplitude_control.item()
            print(f"    Amplitude: {amp_control:.4f}")
            
            # Verificar interferência
            int_strength = model.interference.interference_strength.item()
            print(f"    Interferência: {int_strength:.4f}")
            
    except Exception as e:
        print(f"    Verificação: {e}")

def comprehensive_test(model, dataset):
    """Teste abrangente do modelo otimizado"""
    print("\n🔬 TESTE COMPREENSIVO:")
    
    test_cases = [
        ("quantum wave function superposition physics", 1),
        ("classical mechanics newton laws motion", 0),
        ("schrodinger equation quantum evolution", 1),
        ("simple short text example", 0),
        ("quantum entanglement bell correlation", 1),
        ("thermodynamics heat entropy classical", 0),
        ("wave particle duality interference quantum", 1),
        ("electromagnetism maxwell equations classical", 0)
    ]
    
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        
        for text, expected in test_cases:
            wave = dataset.text_to_physical_wave(text).unsqueeze(0)
            output = model(wave)
            
            prediction = torch.argmax(output).item()
            confidence = F.softmax(output, dim=1).max().item()
            
            status = "✓" if prediction == expected else "✗"
            if prediction == expected:
                correct += 1
            total += 1
            
            print(f"  {status} '{text[:40]}...'")
            print(f"     → Confiança: {confidence:.4f}")
            print(f"     → Previsto: {prediction} (esperado: {expected})")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Performance: {correct}/{total} = {accuracy:.4f}")

def main():
    """Função principal otimizada"""
    print("🧪 GROK-Ω FÍSICO - VERSÃO OTIMIZADA")
    print("=" * 60)
    
    # Treinamento otimizado
    model, dataset = optimized_training()
    
    # Teste abrangente
    comprehensive_test(model, dataset)
    
    print("\n✅ BENCHMARK OTIMIZADO CONCLUÍDO!")
    print("   • Física avançada implementada")
    print("   • Dataset balanceado")
    print("   • Otimização agressiva")
    print("   • Múltiplos Hamiltonianos")
    print("   • Controles físicos aprendíveis")

if __name__ == "__main__":
    main()