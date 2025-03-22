import asyncio
from assistant_voice_server.server import main
import platform

# Check if the operating system is Windows and set the event loop policy
if platform.system() == "Windows":
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
if __name__ == "__main__":
    asyncio.run(main())