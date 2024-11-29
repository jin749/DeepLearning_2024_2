import torch
import torch.nn.functional as F
import random
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from dassl.data.data_manager import build_data_loader, build_transform
from dassl.data.datasets.base_dataset import DatasetBase
from trainers.coop import load_clip_to_cpu, TextEncoder
from tqdm import tqdm
from trainers.zsclip import CUSTOM_TEMPLATES
from clip import clip

class FewShotByClustering:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.clip_model = load_clip_to_cpu(self.cfg).to(self.device)
        self.encode_image = self.clip_model.encode_image
        self.encode_text = self.clip_model.encode_text

    @torch.no_grad()
    def generate_fewshot_dataset(
        self, data_source, num_shots
    ):
        """Generate a few-shot dataset by clustering (typically for the training set).

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
        """
        print(f"Creating a {num_shots}-shot dataset by clustering")

        self.lab2cname_mapping, self.classnames = DatasetBase.get_lab2cname(data_source)

        train_loader = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=data_source,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=True),
                is_train=True,
                dataset_wrapper=None
                )
        
        dataset = []

        indices = self.sample_from_cluster(train_loader, num_shots)
        for idx in indices:
            dataset.append(data_source[idx])

        return dataset
    
    @torch.no_grad()
    def sample_from_cluster(self, data_loader, num_shots):
        ground_labels, image_features = self.get_labels_and_features(data_loader)
        text_features = self.get_text_features(ground_labels.tolist())

        option = 3
        cos_sim_distance = True
        if option == 0:
            # method0: use only image features
            print("Using only image features")
            features = image_features
        elif option == 1:
            # method1: concatenate image and text features
            print("Concatenating image and text features")
            features = torch.cat((image_features, text_features), dim=1)
        elif option == 2:
            # method2: average image and text features
            print("Averaging image and text features")
            features = (image_features + text_features) / 2
        elif option == 3:
            # method3: average image and class-buided text features
            print("Averaging image and class-buided text features")
            target_text_features = self.get_text_features(list(range(0, len(self.classnames))))
            logits = self.clip_model.logit_scale.exp() * image_features @ target_text_features.t()
            probs = F.softmax(logits, dim=-1)
            weighted_text_features = probs @ target_text_features
            features = (image_features + weighted_text_features) / 2

        if cos_sim_distance:
            features = features / features.norm(dim=-1, keepdim=True)

        n_clusters = num_shots * len(self.classnames)
        indices = self.get_k_means_centroids(features, n_clusters, seed=self.cfg.SEED)

        return indices

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output


    def get_k_means_centroids(self, embedding, n_clusters: int, seed: int) -> list:
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

        print(f"Running KMeans with {n_clusters} clusters...", end="")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(embedding)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        centroids_points_idxs = self.find_closest_points(embedding, labels, centroids)
        print("done.")

        return centroids_points_idxs

    def find_closest_points(self, embeddings, labels, centroids):
        closest_points_idxs = []
        for label, centroid in enumerate(centroids):
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            points_in_same_cluster = np.where(labels == label)[0]
            
            n = 1
            while True: 
                closest_point_idx = self.nth_smallest_index(distances, n)
                if closest_point_idx not in points_in_same_cluster:
                    n += 1
                else:
                    closest_points_idxs.append(closest_point_idx)
                    break

        return closest_points_idxs


    def nth_smallest_index(self, arr, n):
        if len(arr) < n:
            raise ValueError("Array must have at least n elements.")
        
        # Add a small value based on the index to make elements unique
        tiny_values = np.arange(len(arr)) * 1e-10
        unique_arr = arr + tiny_values
        
        # Sort the unique array
        sorted_idxs = np.argsort(unique_arr)
        
        nth_index = sorted_idxs[n-1]
        
        return nth_index


    def get_labels_and_features(self, data_loader):
        print("Getting labels and image features...")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
                input_batch = batch["img"].to(self.device)
                ground_labels_batch = batch["label"].to(self.device)

                image_features_batch = self.encode_image(input_batch)
                #image_features_batch = (image_features_batch / image_features_batch.norm(dim=-1, keepdim=True))

                # image_features_batch.to("cpu")
                # ground_labels_batch.to("cpu")

                if batch_idx == 0:
                    ground_labels_full = ground_labels_batch
                    image_features_full = image_features_batch
                else:
                    ground_labels_full = torch.cat((ground_labels_full, ground_labels_batch), dim=0)
                    image_features_full = torch.cat((image_features_full, image_features_batch), dim=0)

        return ground_labels_full, image_features_full

    def get_text_features(self, labels):
        print("Getting text features...")
        classnames = [self.lab2cname_mapping[label] for label in labels]
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts[:5]} ... = {len(prompts)}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        
        with torch.no_grad():
            text_features = self.encode_text(prompts)
            #text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features