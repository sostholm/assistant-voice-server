import asyncio
from assistant_voice_server.server import main
import platform
from asyncio import WindowsSelectorEventLoopPolicy
# Check if the operating system is Windows and set the event loop policy
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
if __name__ == "__main__":
    asyncio.run(main())