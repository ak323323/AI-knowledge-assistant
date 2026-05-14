import os
import json
from datetime import datetime


class DocumentRegistry:
    """
    Central registry for all uploaded documents.

    Responsibilities:
    -----------------
    - Track uploaded files
    - Prevent duplicates
    - Store metadata
    - Support document deletion
    - Enable document filtering
    - Provide analytics
    """

    def __init__(self, storage_path="storage"):

        self.storage_path = storage_path

        os.makedirs(storage_path, exist_ok=True)

        self.registry_file = os.path.join(
            storage_path,
            "documents.json"
        )

        self.documents = self.load()

    # =====================================================
    # LOAD REGISTRY
    # =====================================================

    def load(self):

        if not os.path.exists(self.registry_file):

            return {}

        try:

            with open(self.registry_file, "r", encoding="utf-8") as f:

                return json.load(f)

        except Exception as e:

            print("[REGISTRY] Load failed:", e)

            return {}

    # =====================================================
    # SAVE REGISTRY
    # =====================================================

    def save(self):

        try:

            with open(
                self.registry_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    self.documents,
                    f,
                    indent=2
                )

        except Exception as e:

            print("[REGISTRY] Save failed:", e)

    # =====================================================
    # ADD DOCUMENT
    # =====================================================

    def add_document(
        self,
        doc_id,
        metadata
    ):
        """
        Register uploaded document.
        """

        self.documents[doc_id] = {

            "doc_id": doc_id,

            "file_name": metadata.get(
                "file_name",
                "unknown"
            ),

            "document_title": metadata.get(
                "document_title",
                "unknown"
            ),

            "source_path": metadata.get(
                "source_path",
                ""
            ),

            "uploaded_at": metadata.get(
                "uploaded_at",
                datetime.now().isoformat()
            ),

            "chunk_count": metadata.get(
                "chunk_count",
                0
            ),

            "embedding_model": metadata.get(
                "embedding_model",
                "unknown"
            ),

            "status": "indexed"
        }

        self.save()

        print(
            f"[REGISTRY] Added: "
            f"{metadata.get('file_name')}"
        )

    # =====================================================
    # REMOVE DOCUMENT
    # =====================================================

    def remove_document(self, doc_id):

        if doc_id not in self.documents:

            return False

        removed = self.documents.pop(doc_id)

        self.save()

        print(
            f"[REGISTRY] Removed: "
            f"{removed.get('file_name')}"
        )

        return True

    # =====================================================
    # CHECK EXISTENCE
    # =====================================================

    def document_exists(self, doc_id):

        return doc_id in self.documents

    # =====================================================
    # GET ALL DOCUMENTS
    # =====================================================

    def get_all_documents(self):

        return list(self.documents.values())

    # =====================================================
    # GET DOCUMENT
    # =====================================================

    def get_document(self, doc_id):

        return self.documents.get(doc_id)

    # =====================================================
    # DOCUMENT COUNT
    # =====================================================

    def count(self):

        return len(self.documents)

    # =====================================================
    # CLEAR REGISTRY
    # =====================================================

    def clear(self):

        self.documents = {}

        self.save()

        print("[REGISTRY] Cleared")