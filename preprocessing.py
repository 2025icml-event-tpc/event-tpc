from typing import Union, Sequence
import os
import json


class EventTPTokenizer:
    def __init__(self, events: Sequence[str], unk_token: str = "UNK", pad_token: str = "PAD", pred_token: str = "PRED"):
        self.events = events
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.pred_token = pred_token

        self.tokens = [self.pad_token, self.unk_token, self.pred_token] + self.events
        self.token_id_map = dict(zip(self.tokens, range(len(self.tokens))))

        self.unk_token_id = self.token_id_map[self.unk_token]
        self.pad_token_id = self.token_id_map[self.pad_token]
        self.pred_token_id = self.token_id_map[self.pred_token]

    def get_token_id(self, event: str):
        unk_id = self.token_id_map[self.unk_token]
        return self.token_id_map.get(event, unk_id)
    
    def get_event(self, token_id: int):
        return self.tokens[token_id]

    def encode(self, event_sequence: Sequence[str]):
        return list(map(lambda e: self.token_id_map.get(e), event_sequence))
    
    def decode(self, id_sequence: Sequence[int]):
        return list(map(lambda eid: self.tokens[eid], id_sequence))
    
    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode='w') as f:
            json.dump(dict(events=self.events, unk_token=self.unk_token, pad_token=self.pad_token), f, indent=4)

    @classmethod
    def load(cls, load_path: str):
        with open(load_path, mode='r') as f:
            obj = json.load(f)

        return cls(**obj)