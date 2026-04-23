"""
Script de verificación del entorno PyTorch + CUDA.
Ejecutar con: python test_gpu.py
"""
import torch
import torchvision

print("=" * 60)
print("VERIFICACIÓN DEL ENTORNO DE DEEP LEARNING")
print("=" * 60)

print(f"\n📦 PyTorch version: {torch.__version__}")
print(f"📦 TorchVision version: {torchvision.__version__}")

print(f"\n🖥️  CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🖥️  CUDA version (PyTorch): {torch.version.cuda}")
    print(f"🖥️  Número de GPUs: {torch.cuda.device_count()}")
    print(f"🖥️  GPU actual: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"🖥️  VRAM total: {vram_gb:.2f} GB")
    print(f"🖥️  Compute capability: {props.major}.{props.minor}")

    # Prueba rápida de operación en GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"\n✅ Operación de prueba en GPU exitosa (shape resultado: {z.shape})")
else:
    print("\n⚠️  CUDA no disponible. El entrenamiento se hará en CPU (mucho más lento).")

print("\n" + "=" * 60)
print("✅ Entorno listo para empezar el proyecto")
print("=" * 60)
