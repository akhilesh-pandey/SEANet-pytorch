from pathlib import Path
import torch

#from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    if cuda:
        torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir= '.', save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
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
        print(">>>[{}] loss: {:.2f}/accuracy: {:.2%}".format(mode, loss, acc))
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
        best_acc= 0
        train_acc_list= []
        val_acc_list= []
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            train_acc= self.train(train_data)
            val_acc= self.test(test_data)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            if val_acc > best_acc:
                best_acc= val_acc
                print("Best val acc till now: ",best_acc)
                self.save(ep)
        print("train acc: ", train_acc_list)
        print("val acc: ", val_acc_list)
   

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch, "weight": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, "best_model.pth")
