import torch

def diagnose_training_device(model, train_dataloader):
    # Verificação de GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("\n--- Diagnóstico de Dispositivo ---")
    print(f"GPU Disponível: {use_cuda}")
    
    if use_cuda:
        print("Informações da GPU:")
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
    
    # Move modelo para dispositivo
    model = model.to(device)
    
    # Verifica dispositivo dos parâmetros do modelo
    print("\nLocalização dos Parâmetros do Modelo:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
    
    # Teste com um batch
    print("\nTeste de Transferência de Dados:")
    for batch_input, batch_label in train_dataloader:
        batch_input = {k: v.to(device) for k, v in batch_input.items()}
        batch_label = batch_label.to(device)
        
        print("Dispositivos do Batch:")
        for k, v in batch_input.items():
            print(f"{k}: {v.device}")
        print(f"Labels: {batch_label.device}")
        
        break  # Apenas o 