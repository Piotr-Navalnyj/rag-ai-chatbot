import uuid


class ChatStore:

    def __init__(self, user_id, client):
        self.user_id = user_id
        self.client = client

    def create_chat(self, title="New Chat"):

        chat_id = str(uuid.uuid4())

        self.client.table("chats").insert({
            "id": chat_id,
            "user_id": self.user_id,
            "title": title
        }).execute()

        return chat_id

    def get_chats(self):

        response = (
            self.client
            .table("chats")
            .select("*")
            .eq("user_id", self.user_id)
            .order("updated_at", desc=True)
            .execute()
        )

        return response.data

    def get_messages(self, chat_id):

        chat = (
            self.client
            .table("chats")
            .select("id")
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )

        if not chat.data:
            raise PermissionError(
                "Chat does not belong to this user."
            )

        response = (
            self.client
            .table("messages")
            .select("role, content")
            .eq("chat_id", chat_id)
            .order("created_at")
            .execute()
        )

        return response.data

    def add_message(self, chat_id, role, content):

        chat = (
            self.client
            .table("chats")
            .select("id")
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )

        if not chat.data:
            raise PermissionError(
                "Chat does not belong to this user."
            )

        self.client.table("messages").insert({
            "chat_id": chat_id,
            "role": role,
            "content": content
        }).execute()

        (
            self.client
            .table("chats")
            .update({"updated_at": "now()"})
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )

    def update_title(self, chat_id, title):

        (
            self.client
            .table("chats")
            .update({
                "title": title,
                "updated_at": "now()"
            })
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )

    def delete_chat(self, chat_id):

        (
            self.client
            .table("chats")
            .delete()
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )