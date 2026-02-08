import torch
import sys

print(f"Python : {sys.executable}")
print(f"PyTorch Version : {torch.__version__}")

if torch.cuda.is_available():
    print("✅ SUCCÈS ! GPU DÉTECTÉ")
    print(f"Carte : {torch.cuda.get_device_name(0)}")
    
    # Test de calcul pour être sûr
    x = torch.rand(5, 3).cuda()
    print("Test de tenseur sur GPU : OK")
else:
    print("❌ ÉCHEC : Toujours sur CPU.")
    print("Vérifie que tu as bien lancé ce script avec le bon venv !")