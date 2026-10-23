import os
import json
import logging
import asyncpg
from typing import Optional, List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.is_connected: bool = False
        self._fallback_store: Dict[str, Dict[str, Any]] = {}
        self._audit_store: List[Dict[str, Any]] = []

    async def connect(self):
        """Initializes the asyncpg connection pool and ensures database schema exists."""
        try:
            # First attempt to connect to specified database
            try:
                self.pool = await asyncpg.create_pool(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    min_size=2,
                    max_size=10,
                    command_timeout=30
                )
            except asyncpg.InvalidCatalogNameError:
                # Database does not exist yet -> connect to 'postgres' db and create it
                logger.info(f"Database '{settings.POSTGRES_DB}' does not exist. Creating it...")
                sys_conn = await asyncpg.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database="postgres"
                )
                await sys_conn.execute(f'CREATE DATABASE "{settings.POSTGRES_DB}"')
                await sys_conn.close()
                # Re-attempt connection
                self.pool = await asyncpg.create_pool(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    min_size=2,
                    max_size=10
                )

            self.is_connected = True
            logger.info("Successfully connected to PostgreSQL connection pool.")
            await self.init_schema()

        except Exception as e:
            self.is_connected = False
            self.pool = None
            logger.warning(
                f"PostgreSQL connection failed: {e}. "
                "Operating in safe fallback storage mode. Please verify your credentials in .env to enable persistent PostgreSQL storage."
            )

    async def disconnect(self):
        """Gracefully closes all connections in the pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.is_connected = False
            logger.info("Closed PostgreSQL connection pool.")

    async def init_schema(self):
        """Reads schema.sql and creates tables/indexes if they do not exist."""
        if not self.pool:
            return
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("Database schema initialized and verified.")

    async def execute(self, query: str, *args) -> None:
        """Executes an INSERT, UPDATE, or DELETE query."""
        if self.pool and self.is_connected:
            async with self.pool.acquire() as conn:
                await conn.execute(query, *args)
        else:
            logger.debug(f"Fallback execute: {query}")

    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetches a single row as a dictionary."""
        if self.pool and self.is_connected:
            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(query, *args)
                return dict(record) if record else None
        return None

    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetches all rows as a list of dictionaries."""
        if self.pool and self.is_connected:
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query, *args)
                return [dict(r) for r in records]
        return []

    # Persistence helper methods for document operations
    async def save_document(self, doc_data: Dict[str, Any]) -> None:
        doc_id = doc_data["id"]
        if self.pool and self.is_connected:
            query = """
            INSERT INTO documents (
                id, filename, file_type, file_path, file_size, document_type,
                status, page_count, raw_text, normalized_json, metadata_json, error_message, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
            ON CONFLICT (id) DO UPDATE SET
                document_type = EXCLUDED.document_type,
                status = EXCLUDED.status,
                page_count = EXCLUDED.page_count,
                raw_text = EXCLUDED.raw_text,
                normalized_json = EXCLUDED.normalized_json,
                metadata_json = EXCLUDED.metadata_json,
                error_message = EXCLUDED.error_message,
                updated_at = NOW();
            """
            norm_json = json.dumps(doc_data.get("normalized_json")) if doc_data.get("normalized_json") else None
            meta_json = json.dumps(doc_data.get("metadata_json")) if doc_data.get("metadata_json") else None

            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    doc_id,
                    doc_data.get("filename"),
                    doc_data.get("file_type"),
                    doc_data.get("file_path"),
                    doc_data.get("file_size"),
                    doc_data.get("document_type", "unclassified"),
                    doc_data.get("status", "pending"),
                    doc_data.get("page_count", 1),
                    doc_data.get("raw_text"),
                    norm_json,
                    meta_json,
                    doc_data.get("error_message")
                )
        # Always maintain in-memory fallback mirror
        self._fallback_store[doc_id] = doc_data

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if self.pool and self.is_connected:
            row = await self.fetch_one("SELECT * FROM documents WHERE id = $1", doc_id)
            if row:
                if isinstance(row.get("normalized_json"), str):
                    row["normalized_json"] = json.loads(row["normalized_json"])
                if isinstance(row.get("metadata_json"), str):
                    row["metadata_json"] = json.loads(row["metadata_json"])
                return row
        return self._fallback_store.get(doc_id)

    async def list_documents(
        self,
        document_type: Optional[str] = None,
        filename: Optional[str] = None,
        status: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "DESC",
        json_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        offset = (page - 1) * limit
        
        if self.pool and self.is_connected:
            conditions = ["1=1"]
            params = []
            idx = 1

            if document_type:
                conditions.append(f"document_type = ${idx}")
                params.append(document_type)
                idx += 1
            if filename:
                conditions.append(f"filename ILIKE ${idx}")
                params.append(f"%{filename}%")
                idx += 1
            if status:
                conditions.append(f"status = ${idx}")
                params.append(status)
                idx += 1
            if from_date:
                conditions.append(f"created_at >= ${idx}")
                params.append(from_date)
                idx += 1
            if to_date:
                conditions.append(f"created_at <= ${idx}")
                params.append(to_date)
                idx += 1
            if json_filter:
                conditions.append(f"normalized_json @> ${idx}::jsonb")
                params.append(json.dumps(json_filter))
                idx += 1

            where_clause = " AND ".join(conditions)
            
            # Safe sorting validation
            allowed_sorts = ["created_at", "filename", "document_type", "status", "file_size"]
            order_col = sort_by if sort_by in allowed_sorts else "created_at"
            order_dir = "DESC" if sort_order.upper() == "DESC" else "ASC"

            count_query = f"SELECT COUNT(*) FROM documents WHERE {where_clause}"
            total_records = (await self.fetch_one(count_query, *params))["count"]

            query = f"""
                SELECT id, filename, file_type, file_size, document_type, status,
                       page_count, error_message, created_at, updated_at,
                       normalized_json, metadata_json
                FROM documents
                WHERE {where_clause}
                ORDER BY {order_col} {order_dir}
                LIMIT ${idx} OFFSET ${idx + 1}
            """
            params.extend([limit, offset])
            rows = await self.fetch_all(query, *params)
            for row in rows:
                if isinstance(row.get("normalized_json"), str):
                    row["normalized_json"] = json.loads(row["normalized_json"])
                if isinstance(row.get("metadata_json"), str):
                    row["metadata_json"] = json.loads(row["metadata_json"])

            return {
                "total": total_records,
                "page": page,
                "limit": limit,
                "documents": rows
            }

        # Fallback in-memory search
        docs = list(self._fallback_store.values())
        if document_type:
            docs = [d for d in docs if d.get("document_type") == document_type]
        if filename:
            docs = [d for d in docs if filename.lower() in d.get("filename", "").lower()]
        if status:
            docs = [d for d in docs if d.get("status") == status]
            
        total = len(docs)
        docs.sort(key=lambda x: str(x.get(sort_by, "")), reverse=(sort_order.upper() == "DESC"))
        paged_docs = docs[offset : offset + limit]
        return {
            "total": total,
            "page": page,
            "limit": limit,
            "documents": paged_docs
        }

    async def log_audit(self, doc_id: str, action: str, details: Optional[Dict[str, Any]] = None):
        """Records an audit event."""
        if self.pool and self.is_connected:
            query = "INSERT INTO document_audit (document_id, action, details) VALUES ($1, $2, $3)"
            await self.execute(query, doc_id, action, json.dumps(details or {}))
        else:
            self._audit_store.append({
                "document_id": doc_id,
                "action": action,
                "details": details or {}
            })

db = Database()
