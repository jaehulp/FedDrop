import gc
import torch

from concurrent.futures import ThreadPoolExecutor

from utils.ops import features, hook, compute_cov
from utils.fed.server import Server

class FedCovAvg(Server):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def aggregate_cov_parameters(self):
        assert (len(self.uploaded_models) > 0)

        global_handle = self.global_model.fc.register_forward_hook(hook)
        self.global_model.cuda()
        cov_mean_list = []
        for c in self.join_clients:
            
            loader = c.load_train_data_nonshuffle()
            client_handle = c.model.fc.register_forward_hook(hook)
            c.model.cuda()

            global_output = []
            for i, (xs, ys) in enumerate(loader):
                xs = xs.cuda()
                logits = self.global_model(xs)
                hook_data = features.value.data.view(xs.shape[0], -1).cuda()
                global_output.append(hook_data)
                
            client_output = []
            for i, (xs, ys) in enumerate(loader):
                xs = xs.cuda()
                logits = c.model(xs)
                hook_data = features.value.data.view(xs.shape[0], -1).cuda()
                client_output.append(hook_data)
            client_handle.remove()

            global_output = torch.cat(global_output)
            global_mean = torch.mean(global_output, dim=1, keepdim=True)
            global_output = global_output - global_mean

            client_output = torch.cat(client_output)
            client_mean = torch.mean(client_output, dim=1, keepdim=True)
            client_output = client_output - client_mean

            if self.args_server.svd_proj == True:
                _, _, Vt = torch.linalg.svd(global_output)
                global_output = (Vt[:50] @ global_output.permute(1,0)).T
                _, S, Vt = torch.linalg.svd(client_output)
                client_output = (Vt[:50] @ client_output.permute(1,0)).T

            output_mat = torch.stack([global_output, client_output])
            cov_val = compute_cov(output_mat)

            if self.args_server.svd_proj == True:
                cov_val = abs(cov_val)
            mean = torch.mean(cov_val)
            std = torch.std(cov_val)

            self.logger.info(f'{c.idx} model mean {mean} std {std}')
            cov_mean_list.append(mean.cpu().item())
        global_handle.remove()

        self.logger.info(f'Covariance mean : {np.mean(cov_mean_list)}')

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        if self.args_server.aggregate_cov == 'avg':
            temp = self.args_server.avg_temp
            cov_mean_list = torch.nn.functional.softmax(torch.tensor(cov_mean_list) * temp).tolist()
            self.logger.info(f'Avg weight : {cov_mean_list}')
            for w, client_model in zip(cov_mean_list, self.uploaded_models):
                self.add_parameters(w, client_model)
        
        if self.args_server.aggregate_cov == 'drop':
            num_drop = self.args_server.num_drop
            index = np.argsort(cov_mean_list)
            weights = np.array(self.uploaded_weights)
            weights[index[:num_drop]] = 0
            weights = weights / weights.sum()
            for idx in index[num_drop:]:
                self.add_parameters(weights[idx], self.uploaded_models[idx])
        

    def train(self):

        for epoch in range(1, self.args_server.epochs+1):
            self.select_clients(epoch)
            self.set_client_batch()

            with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:

                processes = []
                for gpu_id, clients in enumerate(self.client_batches):

                    process = executor.submit(
                        self.train_clients_on_gpu,
                        clients, gpu_id
                    )
                    processes.append(process)

                for process in processes:
                    try:
                        process.result()  # Ensure execution and catch runtime exceptions
                    except Exception as e:
                        print(f"Thread execution failed: {e}")
               
            self.receive_models()
            with torch.no_grad():
                self.aggregate_cov_parameters()

            self.save_train_result(epoch)

            self.evaluate(epoch)
            if (epoch % self.args.save_interval == 0) or (epoch % self.args.save_interval == (self.args.save_interval - 1)):
                self.save_global_model(epoch)
                        
            self.del_client_model()
            gc.collect()
            torch.cuda.empty_cache()

