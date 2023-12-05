import Camera_3

import AI_3
import torch as t
import torch.version

print(t.__version__)

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("Cuda aktif")
else:
    print("Cuda aktif değil")
print(torch.version.cuda)

c = Camera_3.Cam()

# AI nesnesini ve modelin kendisini yükle

ai = AI_3.Model(""r"C:\Users\necip\Desktop\ayak_tanima\ai_model\weights\best.pt""") #Buraya ağırlık dosyasının dosya yolu olacak.

c.read_with_ai(ai)
