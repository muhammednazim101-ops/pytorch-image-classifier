import torch
import torch.nn as nn
print('building first ai model...')
model=nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,1))
loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)
celsius=torch.tensor([[0],[10],[20],[30],[40],[50],[60],[70],[80],[90],[100]],dtype=torch.float32)
fahrenheit=torch.tensor([[32],[50],[68],[86],[104],[122],[140],[158],[176],[194],[212]],dtype=torch.float32)
print('training model...')
for epoch in range(5000):
    predictions=model(celsius)
    loss=loss_function(predictions,fahrenheit)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1)%1000==0:
        print(f'round {epoch+1}/5000 - loss: {loss.item():.2f}')
print('training complete!')
test_celsius=torch.tensor([[55]],dtype=torch.float32)
prediction=model(test_celsius)
print('question: what is 55c in fahrenheit?')    
print(f'AI prediction:{prediction.item():.1f}f')
print('actual answer: 77.0f')