import torch
import matplotlib.pyplot as plt


words= open('names.txt','r').read().splitlines()
a = max(len(w) for w in words)

b ={}

for w in words:
    chs = ['<S>']+list(w)+['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram =(ch1,ch2)
        b[bigram] = b.get(bigram,0) + 1
     
# print(sorted(b.items(),key=lambda kv: kv[1])) #sort by count 


N = torch.zeros((27,27), dtype=torch.int32) # 26 alpha + start and end

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # mappinng from char to index a:0
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

#--- 28 x 28 matrices containing counts of bigram
for w in words:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] +=1 

# make the N pretty

# plt.figure(figsize=(26,26))
# plt.imshow(N,cmap="Blues")

# for i in range(27):
#     for j in range(27):
#         chstr = itos[i]+ itos[j]
#         plt.text(j,i,chstr, ha="center", va="bottom", color ="gray")
#         plt.text(j,i,N[i,j].item(),ha="center", va="top", color="gray")

# plt.axis("off")
# plt.show()

P= N.float()
P /=P.sum(1,keepdim=True) # broadcasting (27,27)/(27,1)

g= torch.Generator().manual_seed(2147483647)

# print(P)
for i in range(10):  
    ix =0
    out =[]
    while True:
        p =P[ix] 
        ix =torch.multinomial(p,num_samples=1,replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix ==0: # end token
            break
    
    print(''.join(out)) 
    

