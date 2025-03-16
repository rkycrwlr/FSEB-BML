import torch
import torch.nn as nn
from tqdm import tqdm

def fgsm_attack(model, image, label, epsilon=0.001, thresh=0.7, nsteps=1000):
    
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    out_prob = torch.softmax(output, dim=1)
    

    print(f"Original Probability of gt class: {out_prob[0,1-label.item()]:.2f}")
    
    c = 0
    while out_prob[0,label.item()] < thresh and c < nsteps:
        loss = nn.functional.cross_entropy(output, label)
        
        # Backpropagate to get gradient
        model.zero_grad()
        loss.backward()
        
        # Collect the sign of the gradient
        image.data = image.data - epsilon * image.grad.sign()
        image.data = torch.clamp(image.data, 0, 1)
        image.grad.zero_()
        
        c+=1
        output = model(image)
        out_prob = torch.softmax(output, dim=1)

    print(f"Step Number: {c}")
    print(f"Probability of target class: {out_prob[0,label.item()]:.2f}")
    return image.detach()

def optimize_pixels(model, image, target_label, device, alpha=0.01, num_steps=100, threshold=None, epsilon=1, disable_tqdm=False):
    model.eval() 
    
    x_t = image.clone().detach().to(device)
    
    target = torch.tensor([target_label], dtype=torch.long).to(device)

    probs_target = []
    
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(num_steps), disable=disable_tqdm):
        x_t.requires_grad = True 

        output = model(x_t)
        model.zero_grad()
        loss = criterion(output, target)        
        loss.backward()
        
        x_adv = (x_t - alpha * x_t.grad.data.sign()).detach()
        eta = torch.clamp(x_adv - image, -epsilon, epsilon)
        x_t = torch.clamp(image + eta, 0, 1).detach()

        prob_target = nn.functional.softmax(output, dim=-1)[0, target_label]
        probs_target.append(prob_target.item())
        if threshold and prob_target > threshold:
            break

    return x_t, probs_target