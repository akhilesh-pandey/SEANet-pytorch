from torch import nn


class Aggregate:
    def __init__(self, aggregate_factor):
        self.aggregate_factor= aggregate_factor
    
    def aggregate(self, x):
        b, c, h, w = x.size()
        res= x.reshape(b, c//self.aggregate_factor, self.aggregate_factor, h, w).sum(2)
        return res

class SELayer(nn.Module):
    def __init__(self, channel, reduction= 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        #print("\n______SELAYER____\n")
        b, c, h, w = x.size()
        #print("X shape: ",x.size())
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #print("y shape: ",y.size())
        y= x * y
        #print("X*y shape: ",y.size())
        #k= 4
        #res= y.reshape(b, c//k, k, h, w).sum(2)
        #return res
        return y
