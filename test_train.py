from model import Base
from utils import RadarDataSet
import torch.utils.data as tud
import torch
from tqdm import tqdm
import torch.nn as nn



def train(datapath = './data'):
    BATCH_SIZE = 8
    LEARN_RATE = 1e-4
    EPOCHS = 1000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = RadarDataSet(datapath)
    dataloader = tud.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    total_step = len(dataloader)

    model = Base(channel_num=1).to(device)

    loss_fn = nn.MSELoss()
    optimal = torch.optim.Adam(model.parameters(),lr=LEARN_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        for i,data in tqdm(enumerate(dataloader),total=total_step,desc=f'Epoch {epoch}/{EPOCHS}'):
            inputdata = data['sigData'].to(device)
            ref = data['refSig'].to(device)
            profile = data['profile'].to(device)

            output = model(inputdata,ref)

            loss = loss_fn(profile,output)

            total_loss += loss.item()
            optimal.zero_grad()
            loss.backward()
            optimal.step()
        
        avg_loss = total_loss/total_step
        tqdm.write(f'Average Loss: {avg_loss:.4f}')



if __name__ == '__main__':
    train()