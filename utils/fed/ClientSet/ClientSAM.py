from utils.ops import accuracy, AverageMeter
from utils.sam import SAM, ASAM
from utils.fed.client import Client

class ClientSAM(Client):

    def __init__(self, args, idx: int, output_dir, datapath):
        super().__init__(args, idx, output_dir, datapath)
    
    def train(self):

        acc1_meter, acc5_meter, loss_meter = AverageMeter(), AverageMeter(), AverageMeter()

        train_loader = self.load_train_data()
        self.train_samples = len(train_loader)
        self.model.train()
        self.model.to(self.device)

        if self.args.sam.minimizer == 'sam':
            minimizer = SAM(self.optimizer, self.model, self.args.sam.rho, self.args.sam.eta)
        elif self.args.sam.minimizer == 'asam':
            minimizer = ASAM(self.optimizer, self.model, self.args.sam.rho, self.args.sam.eta)

        for epoch in range(1, self.local_epochs+1):
            for i, (xs, ys) in enumerate(train_loader):
                self.optimizer.zero_grad()
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                logits = self.model(xs)
                acc1, acc5 = accuracy(logits, ys, topk=(1,5))
                loss = self.loss(logits, ys)
                loss.backward()

                minimizer.ascent_step()
                loss.backward()
                minimizer.descent_step()

                acc1_meter.update(acc1), acc5_meter.update(acc5), loss_meter.update(loss)

        self.train_result = [acc1_meter.result().detach().cpu().numpy(), acc5_meter.result().detach().cpu().numpy(), loss_meter.result().detach().cpu().numpy()]

        if self.args.save_client:
            if (self.global_epoch % self.save_epoch == (self.save_epoch - 1)) or (self.global_epoch % self.save_epoch == 0):
                self.save_client_model()
