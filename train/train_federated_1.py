import torch
from train_single import NCFTrainer, NCFTest
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from server_model import ServerNeuralCollaborativeFiltering
import copy
from phe import paillier
import time
import numpy as np
class Utils:
	def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
		self.epoch = 0
		self.num_clients = num_clients
		self.local_path = local_path
		self.server_path = server_path

	def load_pytorch_client_model(self, local_path):
		return torch.jit.load(local_path)

	def get_user_models(self, loader):
		models = []
		for client_id in range(self.num_clients):
			models.append({'model':loader(self.local_path+"dp"+str(client_id)+".pt")})
		return models

	def get_previous_federated_model(self):
		self.epoch += 1
		return torch.jit.load(self.server_path+"server"+str(self.epoch-1)+".pt")

	def save_federated_model(self, model):
		torch.jit.save(model, self.server_path+"server"+str(self.epoch)+".pt")


def federate(utils,client_dict):
	public_key, private_key = paillier.generate_paillier_keypair()
	client_models = utils.get_user_models(utils.load_pytorch_client_model)
	server_model = utils.get_previous_federated_model()
	if len(client_models) == 0:
		utils.save_federated_model(server_model)
		return
	n = len(client_models)

	cnt = 0
	# loss_to_id={}
	# client_loss=[]
	# for k,i in client_dict.items():
	# 	loss_to_id[i]=k
	# 	client_loss.append(i)
	# client_loss.sort()
	# sort_threshold=4
	# if len(client_models)<=sort_threshold:
	# 	client_idx_filtered=range(sort_threshold)
	# else:
	# 	client_idx_filtered=[]
	# 	for i in range(sort_threshold):
	# 		client_idx_filtered.append(loss_to_id[client_loss[i]])


	# server_new_dict = copy.deepcopy(client_models[client_idx_filtered[0]]['model'].state_dict())
	server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
	for i in range(1, len(client_models)):
		# if i not in client_idx_filtered:
		# 	continue
		client_dict = client_models[i]['model'].state_dict()
		for k in client_dict.keys():
			server_new_dict[k] = server_new_dict[k] / n
			server_param_list = server_new_dict[k].cpu().flatten().tolist()
			#加密
			# server_encrypted_param = [public_key.encrypt(val) for val in server_param_list]
			client_dict[k] = client_dict[k] / n

			client_param_list = client_dict[k].cpu().flatten().tolist()
			#加密
			# client_encrypted_param = [public_key.encrypt(val) for val in client_param_list]
			# print("cli:", len(client_param_list))
			# print("sever:", len(server_param_list))
			result = np.array(client_param_list)
			np.savetxt(r"output_client{}.txt".format(cnt), result)
			cnt += 1
			new_server_param_list = []
			#未加密聚合
			for i in range(len(client_param_list)):
				temp = server_param_list[i] + client_param_list[i]
				new_server_param_list.append(temp)
			#加密聚合
			# for i in range(len(server_encrypted_param)):
			#   temp=server_encrypted_param[i] + client_encrypted_param[i]
			#   new_server_param_list.append(temp)
			# server_decrypted_param_list = [private_key.decrypt(enc_param) for enc_param in new_server_param_list]
			# print(len(new_server_param_list))
			server_new_dict[k] = torch.tensor(new_server_param_list).reshape(server_model.state_dict()[k].shape)
		# print(server_new_dict[k].shape, 5)
	server_model.load_state_dict(server_new_dict)
	utils.save_federated_model(server_model)

# def federate(utils):
#     public_key, private_key = paillier.generate_paillier_keypair()
#     client_models = utils.get_user_models(utils.load_pytorch_client_model)
#     server_model = utils.get_previous_federated_model()
#     if len(client_models) == 0:
#       utils.save_federated_model(server_model)
#       return
#     server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
#
#     # print("Starting encryption loop")
#     # # start_time = time.time()
#     # total_iterations = len(server_new_dict)
#     # print("Total iterations:", total_iterations)
#
#     for i in range(1, len(client_models)):
#       # print(f"Aggregating encrypted parameters for client {i}")
#       client_dict = client_models[i]['model'].state_dict()
#       weight_name="output_logits.weight"
#       # shape=client_models[i]['model'].state_dict()[weight_name].shape
#       # param_list=client_dict[weight_name]
#       print(client_models[i]['model'].state_dict()[weight_name])
#       param_list = client_dict[weight_name].cpu().flatten().tolist()
#       encrypted_param = [public_key.encrypt(val) for val in param_list]
#       server_new_dict1 = [public_key.encrypt(0.0) for _ in range(len(encrypted_param))]
#       L=[]
#
#
#     for i in range(len(server_new_dict1)):
#       sum=encrypted_param[i]+server_new_dict1[i]
#       L.append(sum)
#
#
#     print(sum,"finished....")
#     decrypted_param_list = [private_key.decrypt(enc_param) for enc_param in L]
#     print(param_list)
#     print(decrypted_param_list)
#
#
#     decrypted_param = torch.FloatTensor(decrypted_param_list)
#     # print(server_new_dict1)
#     # print(L)
#     # print(decrypted_param)
#     # print(client_dict[weight_name])
#     print(len(server_new_dict1))
#     print(len(client_models))
#     # print(client_models)
#     # print(type(client_dict[weight_name]))
#     # print(type(decrypted_param))

class FederatedNCF:
	def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[1, 5], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128, latent_dim=32, seed=0):
		random.seed(seed)
		self.ui_matrix = ui_matrix
		self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
		self.num_clients = num_clients
		self.latent_dim = latent_dim
		self.user_per_client_range = user_per_client_range
		self.mode = mode
		self.aggregation_epochs = aggregation_epochs
		self.local_epochs = local_epochs
		self.batch_size = batch_size
		self.clients = self.generate_clients()
		self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]
		self.utils = Utils(self.num_clients)
		self.client_dict=None
	def generate_clients(self):
		start_index = 0
		clients = []
		for i in range(self.num_clients):
			users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
			clients.append(NCFTrainer(self.ui_matrix[start_index:start_index+users], epochs=self.local_epochs, batch_size=self.batch_size))
			start_index += users
		return clients

	def single_round(self, epoch=0, first_time=False):
		single_round_results = {key:[] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
		bar = tqdm(enumerate(self.clients), total=self.num_clients)
		loss_dict={}
		for client_id, client in bar:
			results = client.train(self.ncf_optimizers[client_id])
			loss_dict[client_id]=results["loss"]
			for k,i in results.items():
				single_round_results[k].append(i)
			printing_single_round = {"epoch": epoch}
			for k,i in single_round_results.items():
				printing_single_round.update({k:round(sum(i)/len(i), 4) for k,i in single_round_results.items()})
			model = torch.jit.script(client.ncf.to(torch.device("cpu")))
			torch.jit.save(model, "./models/local/dp"+str(client_id)+".pt")
			bar.set_description(str(printing_single_round))
		bar.close()
		return loss_dict

	def extract_item_models(self):
		for client_id in range(self.num_clients):
			model = torch.jit.load("./models/local/dp"+str(client_id)+".pt")
			item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
			item_model.set_weights(model)
			item_model = torch.jit.script(item_model.to(torch.device("cpu")))
			torch.jit.save(item_model, "./models/local_items/dp"+str(client_id)+".pt")

	def train(self):
		first_time = True
		server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
		server_model = torch.jit.script(server_model.to(torch.device("cpu")))
		torch.jit.save(server_model, "./models/central/server"+str(0)+".pt")
		for epoch in range(self.aggregation_epochs):
			server_model = torch.jit.load("./models/central/server"+str(epoch)+".pt", map_location=self.device)
			_ = [client.ncf.to(self.device) for client in self.clients]
			_ = [client.ncf.load_server_weights(server_model) for client in self.clients]
			self.client_dict=self.single_round(epoch=epoch, first_time=first_time)
			first_time = False
			self.extract_item_models()
			federate(self.utils,self.client_dict)

# if __name__ == '__main__':
# 	dataloader = MovielensDatasetLoader()
#
# 	fncf = FederatedNCF(dataloader.ratings, num_clients=2, user_per_client_range=[1, 10], mode="ncf", aggregation_epochs=28, local_epochs=10, batch_size=128)
#
#   fncf.train()
if __name__ == '__main__':
	dataloader = MovielensDatasetLoader()

	fncf = FederatedNCF(dataloader.ratings, num_clients=5, user_per_client_range=[1, 10], mode="ncf", aggregation_epochs=1, local_epochs=10,
											batch_size=128)

	start_time = time.time()
	fncf.train()
	end_time = time.time()

	training_time = end_time - start_time
	print("Training time: ", training_time)



