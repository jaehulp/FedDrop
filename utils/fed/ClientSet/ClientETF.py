from utils.ops import accuracy, AverageMeter
from utils.fed.client import Client

class ClientETF(Client):

    def __init__(self, args, idx: int, output_dir, datapath):
        super().__init__(args, idx, output_dir, datapath)
    
    def train(self):

        acc1_meter, acc5_meter, loss_meter = AverageMeter(), AverageMeter(), AverageMeter()

        train_loader = self.load_train_data()
        self.train_samples = len(train_loader)
        self.model.train()
        self.model.to(self.device)

        for epoch in range(1, self.local_epochs+1):
            for i, (xs, ys) in enumerate(train_loader):
                self.optimizer.zero_grad()
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                logits = self.model(xs)
                acc1, acc5 = accuracy(logits, ys, topk=(1,5))
                loss = self.loss(logits, ys)
                loss.backward()
                self.optimizer.step()

                acc1_meter.update(acc1), acc5_meter.update(acc5), loss_meter.update(loss)

        self.train_result = [acc1_meter.result().detach().cpu().numpy(), acc5_meter.result().detach().cpu().numpy(), loss_meter.result().detach().cpu().numpy()]

        if self.args.save_client:
            if (self.global_epoch % self.save_epoch == (self.save_epoch - 1)) or (self.global_epoch % self.save_epoch == 0):
                self.save_client_model()
