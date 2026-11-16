from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from wordllm_doc import WordLLMDocument

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once, on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class Retriever:
    def __init__(self, path: str | Path, doc:WordLLMDocument) -> None:
        self.path = Path(path)
        self.index_path = self.path.with_suffix(".npy")
        self.doc_path = self.index_path.with_suffix(".json")
        self.model_name = MODEL_NAME
        self.matrix: np.ndarray | None = None
        self.doc = doc
        self._load()

    # ---- persistence ----------------------------------------------------

    def _load(self) -> None:
        if not (self.doc_path.exists()):
            return
        if self.doc.embedding_model != MODEL_NAME:
            # Vectors from a different model are meaningless here; start fresh.
            return
        try:
            self.matrix = np.load(self.index_path)
        except:
            #If matrix cannot be read, forcibly rebuild it
            print("matrix not found.")
            print(f"force rebuilt {self.sync(self.doc,force=True)} blocks")

    def _save(self) -> None:
        np.save(self.index_path,self.matrix)

    # ---- syncing with the WordLLMDocument --------------------------------------

    def sync(self, doc: WordLLMDocument, force=False) -> int:
        """Embed new or changed blocks, drop deleted ones, then save.
        Returns the number of blocks re-embedded. Force always rebuilds every block"""
        blocks = self.doc.blocks.values()
        changed_blocks = [] #If a vector changes, store it here
        changed_positions = [] #If a vector changes, store its row number here
        # Find blocks whose content hash differs from the stored one.
        for i,block in enumerate(blocks):
            hash,changed = block.refresh_hash()
            if changed or force:
                changed_blocks.append(block)
                changed_positions.append(i)
            
        assert (len(changed_positions)==len(changed_blocks))

        # Embed the changed blocks in one batch.
        if changed_positions:
            #Save the hashes to deno
            new_encodings = get_model().encode([b.text for b in changed_blocks], normalize_embeddings=True)
            if self.matrix is None:
                self.matrix=new_encodings
                return len(changed_positions)
            
            new_position_2_vecs = dict(zip(changed_positions,new_encodings))

        # Rebuild the matrix in document order.
            num_rows, num_cols = self.matrix.shape
            updated_rows = [(new_position_2_vecs[i] if i in changed_positions else self.matrix[i]) for i in range(num_rows)]
            self.matrix = np.stack(updated_rows)
        return len(changed_positions)
    # ---- searching ------------------------------------------------------

    def search(self, query: str) -> np.ndarray:
        if self.matrix is None:
            return []
        q = get_model().encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q                      # cosine similarity
        return scores