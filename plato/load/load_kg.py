import torch
from torch_geometric.data import Data
import os
import pickle
import numpy as np

class KGDataset():
    def __init__(self, load_from_scratch=False, save=False, embedding_model="ComplEx", load_dir=None):
        self.embedding_model = embedding_model
        self.load_dir = load_dir
        self.data = None
        self.networkx = None
        self.load_edge_set(load_from_scratch, save)
        self.num_entities = len(self.ent2id)
        self.num_relations = len(self.rel2id)

    def create_pyg_object(self):
        print("Creating PyG Object")
        edge_index = [ [ self.ent2id[str(edge[0])] for edge in self.edge_set ], [ self.ent2id[str(edge[2])] for edge in self.edge_set ] ]
        self.data = Data()
        self.data.edge_index = torch.LongTensor(edge_index)
        self.data.edge_attr = torch.LongTensor([ self.rel2id[str(edge[1])] for edge in self.edge_set ])

        entity_path = os.path.abspath(os.path.join(
            self.load_dir, "biokg_%s_entity.npy" % self.embedding_model
        ))
        relation_path = os.path.abspath(os.path.join(
            self.load_dir, "biokg_%s_relation.npy" % self.embedding_model
        ))
        missing_paths = [
            path for path in (entity_path, relation_path) if not os.path.isfile(path)
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Missing pretrained KG embedding file(s):\n"
                + "\n".join(missing_paths)
                + "\nCheck --load_dir and obtain the matching pretrained files, "
                "or export them using the KG pretraining workflow for this dataset. "
                "Embedding rows must follow the IDs in ent2id.pkl and rel2id.pkl. "
                "This repository does not include an embedding generation script."
            )

        print("Loading KG embedding")
        entity_embedding = np.load(entity_path, allow_pickle=False)
        relation_embedding = np.load(relation_path, allow_pickle=False)
        for name, path, embedding, expected_rows in (
            ("entity", entity_path, entity_embedding, len(self.ent2id)),
            ("relation", relation_path, relation_embedding, len(self.rel2id)),
        ):
            if (
                embedding.ndim != 2
                or embedding.shape[0] != expected_rows
                or embedding.shape[1] == 0
            ):
                raise ValueError(
                    f"Invalid {name} embedding shape in {path}: "
                    f"expected ({expected_rows}, embedding_dim > 0), "
                    f"got {embedding.shape}. Check that the embeddings match "
                    "ent2id.pkl and rel2id.pkl."
                )

        self.data.x = torch.tensor(entity_embedding, dtype=torch.float32)
        self.data.relation_embedding = torch.tensor(
            relation_embedding, dtype=torch.float32
        )

    def load_edge_set(self, load_from_scratch=True, save = False):
        print("Loading KG...")
        if not load_from_scratch:
            print("Loading KG from cache")
            print("Loading KG id2ent, ent2id, rel2id, id2rel")
            self.ent2id = pickle.load(open(os.path.join(self.load_dir, "ent2id.pkl"), "rb")) 
            self.rel2id = pickle.load(open(os.path.join(self.load_dir, "rel2id.pkl"), "rb"))
            self.id2ent = pickle.load(open(os.path.join(self.load_dir, "id2ent.pkl"), "rb"))
            self.id2rel = pickle.load(open(os.path.join(self.load_dir, "id2rel.pkl"), "rb"))
            print("Loading indexed edges")
            self.edge_set = pickle.load(open(os.path.join(self.load_dir, "edge_set.pkl"), "rb"))
            self.create_pyg_object()
            return
        else:
            print("ERROR: Trying to load KG from scratch rather than cache!")
            assert(False)
