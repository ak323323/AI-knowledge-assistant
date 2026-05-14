from storage.document_registry import DocumentRegistry


class DocumentService:

    def __init__(self):

        self.registry = DocumentRegistry()

    # =====================================================
    # LIST DOCUMENTS
    # =====================================================

    def list_documents(self):

        return self.registry.get_all_documents()

    # =====================================================
    # GET DOCUMENT
    # =====================================================

    def get_document(self, doc_id):

        return self.registry.get_document(doc_id)

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================

    def delete_document(
        self,
        doc_id,
        vectorstore
    ):

        # Remove vectors/chunks
        vectorstore.delete_document(doc_id)

        # Remove registry entry
        self.registry.remove_document(doc_id)

        return True

    # =====================================================
    # TOTAL DOCUMENTS
    # =====================================================

    def total_documents(self):

        return self.registry.count()