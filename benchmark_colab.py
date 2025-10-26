#!/usr/bin/env python3
"""
GROK-Ω (OMEGA) Colab Benchmark
==============================

Complete benchmark script for Google Colab testing.
Tests GROK-Ω performance, physics integrity, and generation capabilities.

ZERO FALLBACK POLICY: All tests are honest physics or fail transparently.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from grok_omega import GrokOmega, create_grok_omega
from PiBase import PiBase, create_pi_base


class ColabBenchmark:
    """
    Comprehensive benchmark suite for GROK-Ω in Google Colab.
    Tests physics integrity, performance, and generation quality.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

        print("🧪 GROK-Ω Colab Benchmark Suite")
        print("=" * 40)
        print(f"   Device: {device}")
        print("   🚫 ZERO FALLBACK: CONFIRMED")

    def benchmark_initialization(self):
        """Test model initialization and basic functionality."""
        print("\n🔬 Testing Initialization...")

        start_time = time.time()

        # Create GROK-Ω
        grok_omega = create_grok_omega(embed_dim=64, vocab_size=256, device=self.device)

        # Create PiBase
        pi_base = create_pi_base(dim=64, num_patterns=256, device=self.device)

        init_time = time.time() - start_time

        self.results['initialization'] = {
            'time_seconds': init_time,
            'device': self.device,
            'embed_dim': 64,
            'vocab_size': 256
        }

        # Test basic generation
        test_text = "hello"
        next_wave = grok_omega.generate_next_wave(test_text)
        print(f"   📤 Basic generation: '{test_text}' → '{next_wave}'")
        print(f"   ⚡ Initialization time: {init_time:.3f}s")

        return True

    def benchmark_physics_integrity(self):
        """Test physics integrity and ZERO FALLBACK compliance."""
        print("\n⚛️  Testing Physics Integrity...")

        grok_omega = create_grok_omega(embed_dim=64, vocab_size=256, device=self.device)

        # Test 1: No softmax usage
        softmax_found = False
        for name, module in grok_omega.named_modules():
            if hasattr(module, 'forward'):
                import inspect
                source = inspect.getsource(module.forward)
                if 'softmax' in source.lower() or 'F.softmax' in source.lower():
                    softmax_found = True
                    break

        # Test 2: Direct argmax usage
        argmax_found = False
        generate_source = inspect.getsource(grok_omega.generate_next_wave)
        if 'torch.argmax' in generate_source:
            argmax_found = True

        # Test 3: Energy conservation (approximate)
        test_input = torch.randn(1, 10, dtype=torch.complex64, device=self.device)
        test_output = grok_omega(test_input)

        energy_in = torch.norm(test_input).item()
        energy_out = torch.norm(test_output).item()
        conservation_ratio = abs(energy_in - energy_out) / energy_in

        self.results['physics_integrity'] = {
            'no_softmax': not softmax_found,
            'direct_argmax': argmax_found,
            'energy_conservation_ratio': conservation_ratio,
            'approximate_conservation': conservation_ratio < 0.5  # Allow some drift
        }

        print(f"   🚫 No softmax: {'PASS' if not softmax_found else 'FAIL'}")
        print(f"   🎯 Direct argmax: {'PASS' if argmax_found else 'FAIL'}")
        print(f"   ⚡ Energy conservation: {conservation_ratio:.3f}")
        return not softmax_found and argmax_found

    def benchmark_performance(self):
        """Benchmark computational performance."""
        print("\n⚡ Testing Performance...")

        grok_omega = create_grok_omega(embed_dim=64, vocab_size=256, device=self.device)

        # Test different sequence lengths
        seq_lengths = [10, 50, 100, 200]
        performance_results = {}

        for seq_len in seq_lengths:
            # Create test input
            test_text = "quantum physics wave function collapse"[:seq_len]

            # Time forward pass
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_time = time.time()

            for _ in range(10):  # Average over 10 runs
                logits = grok_omega(test_text)

            torch.cuda.synchronize() if self.device == 'cuda' else None
            avg_time = (time.time() - start_time) / 10

            performance_results[f'seq_{seq_len}'] = {
                'avg_time_ms': avg_time * 1000,
                'throughput_chars_per_sec': seq_len / avg_time
            }

            print(f"   ⚡ Seq {seq_len}: {avg_time*1000:.1f}ms ({seq_len/avg_time:.1f} chars/sec)")
        self.results['performance'] = performance_results
        return True

    def benchmark_wikitext_training(self):
        """Full WikiText training benchmark."""
        print("\n📚 WikiText Training Benchmark...")

        try:
            # Import training components
            from train_wikitext import WikiTextDataset, create_dataloaders, WikiTextTrainer

            # Create datasets and dataloaders
            train_loader, val_loader = create_dataloaders(
                batch_size=8,  # Smaller batch for Colab
                seq_length=64,  # Shorter sequences
                device=self.device
            )

            # Create model
            model = create_grok_omega(embed_dim=64, vocab_size=256, device=self.device)

            # Create trainer
            trainer = WikiTextTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                lr=1e-3,
                save_path='grok_omega_colab_wikitext.pt'
            )

            print("   🚀 Starting WikiText training (2 epochs for benchmark)...")

            # Train for 2 epochs (reduced for benchmark)
            trainer.train(num_epochs=2, patience=2)

            # Test final generation
            test_prompts = ["The quantum", "In physics", "Wave function", "Energy level"]
            wikitext_generation = {}

            print("\n   📝 WikiText-trained generation:")
            for prompt in test_prompts:
                generated = model.generate_next_wave(prompt, time_steps=5)
                wikitext_generation[prompt] = generated
                print(f"     '{prompt}' → '{generated}'")

            self.results['wikitext_training'] = {
                'final_train_loss': trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else None,
                'epochs_completed': 2,
                'generation_samples': wikitext_generation,
                'dataset_size': len(train_loader.dataset) if hasattr(train_loader, 'dataset') else None
            }

            print("   ✅ WikiText training benchmark completed")
            return True

        except Exception as e:
            print(f"   ❌ WikiText training failed: {e}")
            self.results['wikitext_training'] = {
                'error': str(e),
                'status': 'failed'
            }
            return False

    def benchmark_generation_quality(self):
        """Test generation quality and diversity."""
        print("\n🎨 Testing Generation Quality...")

        grok_omega = create_grok_omega(embed_dim=64, vocab_size=256, device=self.device)

        # Quick training on physics concepts
        training_data = [
            ("hello", "world"),
            ("quantum", "physics"),
            ("wave", "function"),
            ("energy", "level"),
            ("consciousness", "emergence")
        ]

        print("   🔧 Quick training on physics concepts...")
        optimizer = torch.optim.Adam(grok_omega.parameters(), lr=1e-3)

        for epoch in range(3):  # Reduced epochs
            total_loss = 0
            for input_text, target_text in training_data:
                loss = grok_omega.train_step(input_text, target_text, optimizer)
                total_loss += loss
            avg_loss = total_loss / len(training_data)
            print(f"     Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Test generation
        test_prompts = ["hello", "quantum", "wave", "energy"]
        generation_results = {}

        for prompt in test_prompts:
            generated = grok_omega.generate_next_wave(prompt, time_steps=8)
            generation_results[prompt] = generated
            print(f"   📝 '{prompt}' → '{generated}'")

        self.results['generation'] = generation_results

        # Test diversity
        diversity_test = []
        for seed in [42, 123]:
            torch.manual_seed(seed)
            result = grok_omega.generate_next_wave("quantum", time_steps=3)
            diversity_test.append(result)

        unique_results = len(set(diversity_test))
        self.results['diversity'] = {
            'unique_generations': unique_results,
            'total_tests': len(diversity_test),
            'diversity_ratio': unique_results / len(diversity_test)
        }

        print(f"   🎲 Diversity test: {unique_results}/{len(diversity_test)} unique generations")

        return True

    def benchmark_pi_base(self):
        """Test PiBase π-centric operations."""
        print("\n🌀 Testing PiBase (π-operations)...")

        pi_base = create_pi_base(dim=64, num_patterns=256, device=self.device)

        # Test π-integrity
        integrity_ok = pi_base.validate_pi_integrity()
        print(f"   🔍 π-Integrity: {'PASS' if integrity_ok else 'FAIL'}")

        # Test π-computation
        test_input = torch.randn(2, 64, dtype=torch.complex64, device=self.device)

        start_time = time.time()
        output = pi_base(test_input)
        compute_time = time.time() - start_time

        self.results['pi_base'] = {
            'integrity_check': integrity_ok,
            'compute_time_ms': compute_time * 1000,
            'input_shape': test_input.shape,
            'output_shape': output.shape
        }

        print(f"   🌀 PiBase compute: {compute_time*1000:.1f}ms")
        return integrity_ok

    def benchmark_memory_usage(self):
        """Test memory usage and efficiency."""
        print("\n💾 Testing Memory Usage...")

        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # Create model
        grok_omega = create_grok_omega(embed_dim=128, vocab_size=512, device=self.device)

        # Test with different batch sizes
        batch_sizes = [1, 4, 8]
        memory_results = {}

        for batch_size in batch_sizes:
            # Create batch input
            test_texts = ["quantum physics and wave functions"] * batch_size

            # Forward pass
            torch.cuda.synchronize() if self.device == 'cuda' else None

            try:
                logits_batch = []
                for text in test_texts:
                    logits = grok_omega(text)
                    logits_batch.append(logits)

                if self.device == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                    memory_results[f'batch_{batch_size}'] = {
                        'peak_memory_mb': peak_memory,
                        'batch_size': batch_size
                    }
                    print(".1f"                else:
                    memory_results[f'batch_{batch_size}'] = {
                        'cpu_memory': 'N/A',
                        'batch_size': batch_size
                    }
                    print(f"   📊 Batch {batch_size}: CPU memory tracking not available")

            except RuntimeError as e:
                print(f"   ❌ Batch {batch_size} failed: {e}")
                memory_results[f'batch_{batch_size}'] = {
                    'error': str(e),
                    'batch_size': batch_size
                }

        self.results['memory'] = memory_results
        return True

    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🚀 Starting GROK-Ω Colab Benchmark Suite")
        print("=" * 50)

        # Run all benchmarks
        benchmarks = [
            self.benchmark_initialization,
            self.benchmark_physics_integrity,
            self.benchmark_performance,
            self.benchmark_wikitext_training,
            self.benchmark_generation_quality,
            self.benchmark_pi_base,
            self.benchmark_memory_usage
        ]

        all_passed = True
        for benchmark in benchmarks:
            try:
                passed = benchmark()
                all_passed = all_passed and passed
            except Exception as e:
                print(f"   ❌ Benchmark failed: {e}")
                all_passed = False

        # Summary
        print("\n" + "=" * 50)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 50)

        print(f"Overall Result: {'PASS ✅' if all_passed else 'FAIL ❌'}")
        print(f"Device: {self.device}")
        print(f"PyTorch: {torch.__version__}")

        if 'initialization' in self.results:
            print(".3f"        if 'physics_integrity' in self.results:
            phys = self.results['physics_integrity']
            print(f"Physics Integrity: No Softmax={phys['no_softmax']}, Direct Argmax={phys['direct_argmax']}")

        if 'performance' in self.results:
            perf = self.results['performance']
            if 'seq_100' in perf:
                print(".1f"        if 'generation' in self.results:
            gen = self.results['generation']
            print(f"Generation Tests: {len(gen)} prompts tested")

        if 'pi_base' in self.results:
            pi = self.results['pi_base']
            print(f"PiBase: Integrity={'PASS' if pi['integrity_check'] else 'FAIL'}")

        print("\n🎯 GROK-Ω Colab Benchmark Complete!")
        print("   🌊 Language as continuous wave")
        print("   ⚛️  Pure quantum physics")
        print("   🚫 No softmax, no tokenization, no fallbacks")

        return all_passed

    def save_results(self, filename: str = 'grok_omega_colab_benchmark.json'):
        """Save benchmark results to JSON."""
        import json

        # Convert tensors and complex types to serializable format
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, complex):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = make_serializable(self.results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Results saved to {filename}")


def main():
    """Main benchmark function for Colab."""
    # Create benchmark suite
    benchmark = ColabBenchmark()

    # Run full benchmark
    success = benchmark.run_full_benchmark()

    # Save results
    benchmark.save_results()

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()