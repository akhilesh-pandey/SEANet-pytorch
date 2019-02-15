from pathlib import Path
import torch

from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.manual_seed(9)
    if cuda:
        torch.cuda.manual_seed_all(9)
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir= '.', save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.start_epoch= 0
        self.best_acc1= 0

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        #for data, target in tqdm(data_loader, ncols= 80):
        for data, target in data_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        loss= sum(loop_loss)
        acc= sum(accuracy)/len(data_loader.dataset)
        mode = "train" if is_train else "test"
        #print(f">>>[{mode}] loss: {sum(loop_loss):.2f}/accuracy: {sum(accuracy) / len(data_loader.dataset):.2%}")
        print(f">>>[{mode}] loss: {loss:.2f}/accuracy: {acc:.2%}")
        return loss, acc

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            loss, correct = self._iteration(data_loader)
            return correct

    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, is_train=False)
            return correct

    def loop(self, epochs, train_data, test_data, scheduler=None):
        #best_acc= 0
        train_acc_list= []
        val_acc_list= []
        for ep in range(self.start_epoch, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            train_acc= self.train(train_data)
            best_acc= self.test(test_data)
            train_acc_list.append(train_acc)
            val_acc_list.append(best_acc)
            if best_acc > self.best_acc1:
                self.best_acc1= best_acc
                print("Best val acc till now: ",self.best_acc1)
                self.save(ep+1)
        print("train acc: ", train_acc_list)
        print("val acc: ", val_acc_list)
   

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "best_acc1":self.best_acc1}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "best_model.pth".format(epoch))
    
    def load(self, path="best_model.pth"):
        checkpoint= torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch= checkpoint['epoch']
        self.best_acc1= checkpoint['best_acc1']
        