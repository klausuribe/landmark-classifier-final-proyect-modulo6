"""
train.py
Loop de entrenamiento con validation, checkpointing y logging de métricas.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from pathlib import Path


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Ejecuta una época de entrenamiento. Retorna (loss, accuracy) promedio."""
    model.train()  # Activa dropout y batchnorm en modo training

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # 1. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward pass + optimización
        optimizer.zero_grad()   # Limpia gradientes del batch anterior
        loss.backward()          # Calcula gradientes (backprop)
        optimizer.step()         # Actualiza pesos

        # Métricas
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Evalúa el modelo en validación/test. Retorna (loss, accuracy)."""
    model.eval()  # Desactiva dropout y congela batchnorm

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No calculamos gradientes (ahorra memoria y tiempo)
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, save_path, scheduler=None):
    """
    Loop completo de entrenamiento con early stopping implícito
    (guardamos el modelo con mejor val_loss).

    Returns:
        dict con historiales: {'train_loss', 'train_acc', 'val_loss', 'val_acc'}
    """
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = -1

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Iniciando entrenamiento en {device} por {num_epochs} épocas...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Época {epoch+1}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Val
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Logging
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Guardar mejor modelo (menor val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"✅ Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")

        # Scheduler (si se provee)
        if scheduler is not None:
            scheduler.step(val_loss) if hasattr(scheduler, 'step') else scheduler.step()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Entrenamiento completado en {elapsed/60:.1f} minutos")
    print(f"Mejor época: {best_epoch} con val_loss={best_val_loss:.4f}")
    print(f"{'='*60}")

    return history
