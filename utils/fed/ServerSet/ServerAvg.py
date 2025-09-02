import gc
import torch

from concurrent.futures import ThreadPoolExecutor

from utils.fed.server import Server

class FedAvg(Server):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

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
                self.aggregate_parameters()

            self.save_train_result(epoch)

            self.evaluate(epoch)
            if (epoch % self.args.save_interval == 0) or (epoch % self.args.save_interval == (self.args.save_interval - 1)):
                self.save_global_model(epoch)
                        
            self.del_client_model()
            gc.collect()
            torch.cuda.empty_cache()

