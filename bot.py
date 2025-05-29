import os
import json
import asyncio
from typing import Dict, List

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from openai import AsyncOpenAI

from vector_store import FaissVectorStore
from tools import TOOLS, TOOL_FUNCTIONS

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

bot = Bot(TELEGRAM_TOKEN)
dp = Dispatcher()

# store chat history per user
history: Dict[int, List[dict]] = {}
# lock per user to prevent concurrent handling of multiple messages
locks: Dict[int, asyncio.Lock] = {}
vector_stores: Dict[int, FaissVectorStore] = {}

ALLOWED_USERNAME = "skulabuhov"


def is_allowed(message: types.Message) -> bool:
    return message.from_user and message.from_user.username == ALLOWED_USERNAME


@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    if not is_allowed(message):
        await message.answer("Access denied.")
        return
    await message.answer("Привет! Отправьте сообщение, и я постараюсь помочь.")


def get_vector_store(user_id: int) -> FaissVectorStore:
    store = vector_stores.get(user_id)
    if store is None:
        path = f"data/{user_id}"
        store = FaissVectorStore.load(openai_client, path)
        vector_stores[user_id] = store
    return store


async def add_to_history(user_id: int, role: str, content: str):
    msgs = history.setdefault(user_id, [])
    msgs.append({"role": role, "content": content})
    while len(msgs) > 15:
        old = msgs.pop(0)
        store = get_vector_store(user_id)
        await store.add(old["content"])
        store.save(f"data/{user_id}")


async def prepare_context(user_id: int, query: str) -> List[str]:
    store = get_vector_store(user_id)
    results = await store.search(query)
    return results


@dp.message()
async def handle_message(message: types.Message):
    if not is_allowed(message):
        await message.answer("Access denied.")
        return

    user_id = message.from_user.id
    lock = locks.setdefault(user_id, asyncio.Lock())
    if lock.locked():
        await message.answer("Подожди пока закончу с ответом.")
        return

    async with lock:
        await message.reply("сейчас подумаю...")

        await add_to_history(user_id, "user", message.text)

        msgs = history[user_id].copy()

        context = await prepare_context(user_id, message.text)
        if context:
            msgs.append({
                "role": "system",
                "content": "Дополнительный контекст:\n" + "\n".join(context)
            })

        response = await openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
        )

        reply = ""
        msg = response.choices[0].message
        if msg.tool_calls:
            for call in msg.tool_calls:
                func_name = call.function.name
                args = json.loads(call.function.arguments)
                if func_name in TOOL_FUNCTIONS:
                    result = await TOOL_FUNCTIONS[func_name](**args)
                    await add_to_history(user_id, "tool", result)
                    msgs.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call.id,
                        "name": func_name,
                    })
            # call assistant again with tool results
            response = await openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=msgs,
            )
            msg = response.choices[0].message
            reply = msg.content
        else:
            reply = msg.content

        await add_to_history(user_id, "assistant", reply)
        await message.answer(reply)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
