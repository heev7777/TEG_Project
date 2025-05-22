import asyncio
import logging
import subprocess
import time
import sys
from pathlib import Path
import httpx # For health check

# Assuming your settings will be loaded by the modules when they are imported
# Or, if run_tests.py is the entry point, it might need to load them for itself too
# For now, let's ensure app.core.config defines these for when the server/tests run
# If this script is in the project root:
try:
    from app.core.config import settings # Try to import to get server host/port
    MCP_SERVER_HOST = settings.MCP_SERVER_HOST
    MCP_SERVER_PORT = settings.MCP_SERVER_PORT
except ImportError:
    # Fallback if settings cannot be imported directly by this script (e.g. path issues)
    # This indicates a potential issue with how you structure imports or run scripts
    logger.warning("Could not import settings directly, using default host/port for health check.")
    MCP_SERVER_HOST = "127.0.0.1"
    MCP_SERVER_PORT = 8001 # Must match your mcp_server.py default/config

HEALTH_CHECK_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/health"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def wait_for_server_ready(url: str, timeout_seconds: int = 30, poll_interval: float = 1.0):
    """Polls the server's health check endpoint until it's ready or timeout."""
    logger.info(f"Waiting for MCP server to be ready at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            async with httpx.AsyncClient(timeout=0.5) as client: # Quick timeout for health check
                response = await client.get(url)
            if response.status_code == 200:
                logger.info(f"MCP server at {url} is ready!")
                return True
        except httpx.ConnectError:
            logger.debug(f"MCP server not yet ready (connection error)... retrying in {poll_interval}s")
        except httpx.TimeoutException: # Specific timeout for the health check request itself
            logger.debug(f"MCP server not yet ready (health check timeout)... retrying in {poll_interval}s")
        except Exception as e:
            logger.warning(f"Health check to {url} failed with unexpected error: {e}")
        await asyncio.sleep(poll_interval) # Use asyncio.sleep in async function
    logger.error(f"MCP server at {url} did not become ready within {timeout_seconds} seconds.")
    return False

async def start_mcp_server():
    """Start the MCP server in a separate process."""
    logger.info("Starting MCP server...")
    # Ensure this command correctly points to your Python interpreter and mcp_server module
    # Running as a module from the project root (where run_tests.py is) is generally good.
    server_process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "app.mcp_server",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    logger.info(f"MCP server process started with PID: {server_process.pid}")

    # Wait for server to be ready using health check
    if not await wait_for_server_ready(HEALTH_CHECK_URL):
        # If server didn't become ready, try to get output and terminate
        stdout, stderr = await server_process.communicate()
        logger.error("MCP Server failed to start or become healthy.")
        if stdout:
            logger.error(f"MCP Server STDOUT:\n{stdout.decode(errors='ignore')}")
        if stderr:
            logger.error(f"MCP Server STDERR:\n{stderr.decode(errors='ignore')}")
        server_process.terminate()
        await server_process.wait()
        raise RuntimeError("MCP Server failed to start for testing.")
    
    return server_process

async def run_pytest_suite():
    """Run the pytest test suite."""
    logger.info("Running pytest test suite...")
    # Using pytest to run tests from the 'tests' directory
    # -s: show print statements from tests
    # -v: verbose output
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", "tests/test_product_comparison.py", "-v", "-s",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    print("\n--- Pytest STDOUT ---")
    decoded_stdout = stdout.decode(errors='ignore')
    print(decoded_stdout)
    print("--- End Pytest STDOUT ---")

    if stderr:
        print("\n--- Pytest STDERR ---")
        decoded_stderr = stderr.decode(errors='ignore')
        print(decoded_stderr)
        print("--- End Pytest STDERR ---")

    if process.returncode == 0:
        logger.info("All pytest tests passed!")
        return True
    else:
        logger.error(f"Pytest tests failed with return code {process.returncode}!")
        return False

async def main_test_orchestrator():
    """Orchestrates starting server, running tests, and stopping server."""
    server_process = None
    tests_passed = False
    try:
        server_process = await start_mcp_server()
        tests_passed = await run_pytest_suite()
        
    except Exception as e:
        logger.error(f"Error during test orchestration: {e}", exc_info=True)
    finally:
        if server_process:
            logger.info(f"Stopping MCP server (PID: {server_process.pid})...")
            if server_process.returncode is None: # Check if process is still running
                try:
                    server_process.terminate()
                    await asyncio.wait_for(server_process.wait(), timeout=5.0)
                    logger.info("MCP server terminated.")
                except asyncio.TimeoutError:
                    logger.warning("MCP server did not terminate gracefully after 5s, killing.")
                    server_process.kill()
                    await server_process.wait()
                    logger.info("MCP server killed.")
                except Exception as e_term: # Catch other potential errors during termination
                    logger.error(f"Exception during MCP server termination: {e_term}")
            else:
                logger.info(f"MCP server already exited with code: {server_process.returncode}")
        
        if not tests_passed:
            sys.exit(1) # Exit with error code if tests failed

if __name__ == "__main__":
    asyncio.run(main_test_orchestrator())