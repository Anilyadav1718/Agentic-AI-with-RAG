import json
import sqlite3
import uuid
from datetime import datetime, timezone


# ============================================================
# CONFIG
# ============================================================

DATABASE_PATH = "nexor_ai.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# TIME
# ============================================================

def current_time():
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# COLUMN CHECK
# ============================================================

def column_exists(
    connection,
    table_name,
    column_name,
):
    """
    Check whether a column already exists.

    Used for automatic database migrations.
    """

    columns = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return any(
        column["name"] == column_name
        for column in columns
    )


# ============================================================
# DATABASE INITIALIZATION + MIGRATIONS
# ============================================================

def init_database():
    """
    Initialize Nexor AI SQLite database.

    Also upgrades older database versions automatically.
    """

    with get_connection() as connection:

        # ====================================================
        # CONVERSATIONS TABLE
        # ====================================================

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'assistant',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        # ====================================================
        # MESSAGES TABLE
        # ====================================================

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,

                FOREIGN KEY(conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
            )
            """
        )

        # ====================================================
        # MIGRATION:
        # OLD DATABASE -> ADD metadata_json
        # ====================================================

        if not column_exists(
            connection,
            "messages",
            "metadata_json",
        ):
            connection.execute(
                """
                ALTER TABLE messages
                ADD COLUMN metadata_json TEXT
                """
            )

        # ====================================================
        # MIGRATION:
        # conversations.mode
        # ====================================================

        if not column_exists(
            connection,
            "conversations",
            "mode",
        ):
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN mode TEXT
                NOT NULL DEFAULT 'assistant'
                """
            )

        # ====================================================
        # MIGRATION:
        # conversations.created_at
        # ====================================================

        if not column_exists(
            connection,
            "conversations",
            "created_at",
        ):
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN created_at TEXT
                """
            )

            connection.execute(
                """
                UPDATE conversations
                SET created_at = ?
                WHERE created_at IS NULL
                """,
                (
                    current_time(),
                ),
            )

        # ====================================================
        # MIGRATION:
        # conversations.updated_at
        # ====================================================

        if not column_exists(
            connection,
            "conversations",
            "updated_at",
        ):
            connection.execute(
                """
                ALTER TABLE conversations
                ADD COLUMN updated_at TEXT
                """
            )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE updated_at IS NULL
                """,
                (
                    current_time(),
                ),
            )

        # ====================================================
        # INDEXES
        # ====================================================

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_conversation
            ON messages (
                conversation_id,
                id
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_conversations_updated
            ON conversations (
                updated_at DESC
            )
            """
        )

        connection.commit()


# ============================================================
# CREATE CONVERSATION
# ============================================================

def create_conversation(
    mode="assistant",
    title="New chat",
):
    conversation_id = str(
        uuid.uuid4()
    )

    now = current_time()

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO conversations (
                id,
                title,
                mode,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                title,
                mode,
                now,
                now,
            ),
        )

        connection.commit()

    return conversation_id


# ============================================================
# GET ONE CONVERSATION
# ============================================================

def get_conversation(
    conversation_id,
):
    if not conversation_id:
        return None

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                id,
                title,
                mode,
                created_at,
                updated_at
            FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        ).fetchone()

    if not row:
        return None

    return dict(row)


# ============================================================
# GET CONVERSATIONS
# ============================================================

def get_conversations(
    search_query="",
    limit=5,
):
    """
    Return stored conversations.

    Empty conversations are intentionally hidden.

    Normal sidebar:
        latest 5 conversations

    Search:
        app.py can request a larger limit.
    """

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5

    limit = max(
        1,
        min(limit, 100),
    )

    search_query = (
        search_query or ""
    ).strip()

    with get_connection() as connection:

        if search_query:

            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.mode,
                    c.created_at,
                    c.updated_at
                FROM conversations c

                WHERE EXISTS (
                    SELECT 1
                    FROM messages m
                    WHERE m.conversation_id = c.id
                )

                AND (
                    LOWER(c.title)
                    LIKE LOWER(?)

                    OR EXISTS (
                        SELECT 1
                        FROM messages sm
                        WHERE
                            sm.conversation_id = c.id
                            AND LOWER(sm.content)
                                LIKE LOWER(?)
                    )
                )

                ORDER BY c.updated_at DESC

                LIMIT ?
                """,
                (
                    f"%{search_query}%",
                    f"%{search_query}%",
                    limit,
                ),
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.mode,
                    c.created_at,
                    c.updated_at
                FROM conversations c

                WHERE EXISTS (
                    SELECT 1
                    FROM messages m
                    WHERE m.conversation_id = c.id
                )

                ORDER BY c.updated_at DESC

                LIMIT ?
                """,
                (
                    limit,
                ),
            ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# GET MESSAGES
# ============================================================

def get_messages(
    conversation_id,
):
    if not conversation_id:
        return []

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                conversation_id,
                role,
                content,
                metadata_json,
                created_at

            FROM messages

            WHERE conversation_id = ?

            ORDER BY id ASC
            """,
            (
                conversation_id,
            ),
        ).fetchall()

    messages = []

    for row in rows:

        item = dict(row)

        metadata = {}

        metadata_json = item.get(
            "metadata_json"
        )

        if metadata_json:

            try:
                metadata = json.loads(
                    metadata_json
                )

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                metadata = {}

        messages.append(
            {
                "id": item["id"],
                "conversation_id": (
                    item[
                        "conversation_id"
                    ]
                ),
                "role": item["role"],
                "content": item["content"],
                "created_at": (
                    item["created_at"]
                ),

                # Agent metadata
                "tools": metadata.get(
                    "tools",
                    [],
                ),

                # RAG metadata
                "sources": metadata.get(
                    "sources",
                    [],
                ),

                # Observability metadata
                "metrics": metadata.get(
                    "metrics",
                    {},
                ),
            }
        )

    return messages


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    conversation_id,
    role,
    content,
    tools=None,
    sources=None,
    metrics=None,
):
    """
    Save user/assistant message.

    tools:
        Agent tool events.

    sources:
        RAG sources.

    metrics:
        Agent/RAG observability metrics.
    """

    if not conversation_id:
        return

    metadata = {
        "tools": tools or [],
        "sources": sources or [],
        "metrics": metrics or {},
    }

    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        default=str,
    )

    now = current_time()

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO messages (
                conversation_id,
                role,
                content,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                metadata_json,
                now,
            ),
        )

        # Move conversation to top of Recents.
        connection.execute(
            """
            UPDATE conversations

            SET updated_at = ?

            WHERE id = ?
            """,
            (
                now,
                conversation_id,
            ),
        )

        connection.commit()


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

def update_conversation_title(
    conversation_id,
    title,
):
    if not conversation_id:
        return

    title = (
        title
        or "New chat"
    ).strip()

    if not title:
        title = "New chat"

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE conversations

            SET
                title = ?,
                updated_at = ?

            WHERE id = ?
            """,
            (
                title,
                current_time(),
                conversation_id,
            ),
        )

        connection.commit()


# ============================================================
# UPDATE CONVERSATION MODE
# ============================================================

def update_conversation_mode(
    conversation_id,
    mode,
):
    if not conversation_id:
        return

    if mode not in {
        "assistant",
        "rag",
    }:
        return

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE conversations

            SET
                mode = ?,
                updated_at = ?

            WHERE id = ?
            """,
            (
                mode,
                current_time(),
                conversation_id,
            ),
        )

        connection.commit()


# ============================================================
# DELETE ONE CONVERSATION
# ============================================================

def delete_conversation(
    conversation_id,
):
    if not conversation_id:
        return

    with get_connection() as connection:

        # Explicitly delete messages too.
        #
        # ON DELETE CASCADE should already handle this,
        # but this also protects older database schemas
        # that may not have been created with the FK.

        connection.execute(
            """
            DELETE FROM messages
            WHERE conversation_id = ?
            """,
            (
                conversation_id,
            ),
        )

        connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (
                conversation_id,
            ),
        )

        connection.commit()


# ============================================================
# CLEAR ALL CONVERSATIONS
# ============================================================

def clear_all_conversations():
    """
    Permanently delete ALL chat history.

    This does not delete the SQLite database itself.
    """

    with get_connection() as connection:

        connection.execute(
            """
            DELETE FROM messages
            """
        )

        connection.execute(
            """
            DELETE FROM conversations
            """
        )

        # Reset message AUTOINCREMENT counter.
        try:

            connection.execute(
                """
                DELETE FROM sqlite_sequence
                WHERE name = 'messages'
                """
            )

        except sqlite3.OperationalError:
            pass

        connection.commit()


# ============================================================
# DELETE EMPTY CONVERSATIONS
# ============================================================

def delete_empty_conversations():
    """
    Remove old empty 'New chat' records.

    This fixes databases created by the previous version
    where clicking New Chat immediately inserted a row.
    """

    with get_connection() as connection:

        connection.execute(
            """
            DELETE FROM conversations

            WHERE NOT EXISTS (
                SELECT 1
                FROM messages

                WHERE messages.conversation_id
                    = conversations.id
            )
            """
        )

        connection.commit()


# ============================================================
# GENERATE CHAT TITLE
# ============================================================

def generate_title(
    prompt,
    max_length=45,
):
    """
    Generate a local conversation title.

    No LLM/API call is required.
    """

    if not prompt:
        return "New chat"

    # Remove extra whitespace/newlines.
    title = " ".join(
        prompt.strip().split()
    )

    if not title:
        return "New chat"

    # Keep short prompts unchanged.
    if len(title) <= max_length:
        return title

    shortened = title[
        :max_length
    ]

    # Avoid cutting in the middle of a word.
    if " " in shortened:
        shortened = shortened.rsplit(
            " ",
            1,
        )[0]

    shortened = shortened.rstrip(
        ".,!?;:-"
    )

    if not shortened:
        shortened = title[
            :max_length
        ]

    return (
        shortened
        + "..."
    )