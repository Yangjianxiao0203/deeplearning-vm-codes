import torch
import numpy as np

from Model import Net

import matplotlib.pyplot as plt

def get_train_data():
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return X_tensor, y_tensor

def get_test_data():
    X = np.load('X_test.npy')
    y = np.load('y_test.npy')
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return X_tensor, y_tensor

# 评估模型
def evaluate(model:torch.nn.Module,X_test:torch.tensor,y_test:torch.tensor):
    model.eval()
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model(X_test)
        for i in range(len(y_pred)):
            y_pred_label = torch.argmax(y_pred[i])
            if y_pred_label == y_test[i]:
                correct += 1
            else:
                wrong += 1
    print('correct:{},wrong:{},accuracy:{}'.format(correct,wrong,correct/(correct+wrong)))
    return correct/(correct+wrong)

if __name__ == '__main__':
    X,y = get_train_data()
    print(X.shape)
    print(y)
    
    X_test,y_test = get_test_data()
    
    epoch_num = 100
    batch_size = 128
    input_dim = X.shape[1]
    learning_rate = 1e-3
    number_class = 5
    
    model = Net(input_dim, number_class)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log=[]
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(X.shape[0]//batch_size):
            optim.zero_grad()
            start_idx = batch_index * batch_size
            end_idx = min((batch_index + 1) * batch_size, X.shape[0])  
            x = X[start_idx:end_idx]
            y = y[start_idx:end_idx]
            
            loss = model(x,y)
            loss.backward()
            optim.step()
            
            # watch_loss.append(loss.item())
        break;
    
        print('epoch:{},loss:{}'.format(epoch,np.mean(watch_loss)))
        acc = evaluate(model,X_test,y_test)
        log.append([acc,np.mean(watch_loss)])
    
    torch.save(model.state_dict(),'model.pth')
    print('model saved')
    print(log)
    
    plt.plot(range(len(log)),[l[0] for l in log],label="acc")
    plt.plot(range(len(log)),[l[1] for l in log],label="loss")
    plt.legend()
    plt.savefig('model_performance.png')
    
    
    