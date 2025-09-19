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

P= (N+1).float() # +1 is for model smoothing (laplace)
P /=P.sum(1,keepdim=True) # broadcasting (27,27)/(27,1)

g= torch.Generator().manual_seed(2147483647)

# print(P) 
# prediction
for i in range(10):  
    ix =0
    out =[]
    while True:
        p =P[ix] 
        ix =torch.multinomial(p,num_samples=1,replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix ==0: # end token
            break
    
    # print(''.join(out)) 
    

log_likelihood = 0.0
n=0
for w in words[1:]:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        log_likelihood +=logprob
        # print(f'{ch1}{ch2}:{prob:.4f} {logprob:.4f}')
        n += 1
    

print(f'{log_likelihood =}')
nll = - log_likelihood
print(f'{nll =}')
print(f'{nll/n}')

# create the training set of bigrams(x,y)

xs,ys =[], []

for w in words[1:]:
    chs = ['.']+list(w)+['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(xs,ys)

# forward pass
import torch.nn.functional as F
xenc = F.one_hot(xs,num_classes=27).float()
W= torch.randn((27,27),generator=g, requires_grad=True)
logits = xenc @ W #log counts
counts = logits.exp()
probs = counts / counts.sum(1,keepdim=True)
loss = -probs[torch.arange(5),ys].log().mean()

#backward pass
W.grad = None # set to zero the gradient pytorch
loss.backward()
