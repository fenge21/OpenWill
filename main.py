"""OpenWill - An agent with free will

Usage:
    python main.py                          # Start the agent
    python main.py --cycles 10              # Run 10 cycles
    python main.py --model gpt-4o           # Specify model
    python main.py --provider ollama         # Use Ollama
    python main.py --install-service         # Register as system service (auto-start on boot)
    python main.py --uninstall-service       # Uninstall system service
"""

import argparse
import logging
import sys

from openwill.config import AgentConfig
from openwill.agent import OpenWillAgent
from openwill.tools.recovery import startup_recovery
from openwill.tools.bluegreen import check_and_deploy
from openwill.tools.self_restart import is_another_instance_running, kill_other_instance
from openwill.tools.service import install_as_service, uninstall_service


def setup_logging(level: str = "INFO"):
    """Set up logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="OpenWill - An agent with free will")
    parser.add_argument("--cycles", type=int, default=0, help="Maximum number of cycles (0=infinite)")
    parser.add_argument("--model", type=str, help="LLM model name")
    parser.add_argument("--provider", type=str, help="LLM provider (openai/anthropic/ollama)")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--base-url", type=str, help="API base URL")
    parser.add_argument("--delay", type=float, help="Cycle interval (seconds)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--install-service", action="store_true", help="Register as system service (auto-start on boot)")
    parser.add_argument("--uninstall-service", action="store_true", help="Uninstall system service")
    parser.add_argument("--self-spawned", action="store_true", help=argparse.SUPPRESS)  # Internal flag

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Service management
    if args.install_service:
        if install_as_service():
            print("✅ System service registered. OpenWill will auto-start on boot.")
        else:
            print("❌ Failed to register system service.")
        return

    if args.uninstall_service:
        if uninstall_service():
            print("✅ System service uninstalled.")
        else:
            print("❌ Failed to uninstall system service.")
        return

    # Load configuration
    config = AgentConfig.from_env()

    # Command-line argument overrides
    if args.model:
        config.llm.model = args.model
    if args.provider:
        config.llm.provider = args.provider
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.base_url:
        config.llm.base_url = args.base_url
    if args.delay:
        config.cycle_delay = args.delay
    if args.data_dir:
        config.memory.data_dir = args.data_dir

    # Check required configuration
    if not config.llm.api_key and config.llm.provider != "ollama":
        print("Error: API key is required. Please provide it via one of the following:")
        print("  1. Set environment variable OPENAI_API_KEY or LLM_API_KEY")
        print("  2. Command-line argument --api-key")
        print("  3. Use Ollama: --provider ollama")
        sys.exit(1)

    # ===== Pre-start checks =====

    # 1. Blue-green deployment: check if a new version is pending deployment
    print("Checking for new version pending deployment...")
    deploy_result = check_and_deploy()
    if deploy_result.get("deployed"):
        if deploy_result.get("rollback"):
            print("⚠️ New version failed self-check after deployment, automatically rolled back to previous version")
            for err in deploy_result.get("errors", []):
                print(f"  Error: {err}")
        else:
            print("🚀 Switched to new version!")
            if deploy_result.get("backup_dir"):
                print(f"  Previous version backup: {deploy_result['backup_dir']}")
    print()

    # 2. Code integrity self-check
    print("Running startup self-check...")
    recoveries = startup_recovery()
    if recoveries:
        print(f"⚠️ Detected {len(recoveries)} files needing repair:")
        for r in recoveries:
            status = "✅ Repaired" if r["recovered"] else "❌ Repair failed"
            print(f"  {status} {r['file']}")
        print()

    # 3. Prevent multiple instances
    if is_another_instance_running():
        print("⚠️ Detected another OpenWill instance running")
        if not args.self_spawned:
            print("  To replace the old instance, please stop it first")
            print("  If the old instance is unresponsive, the watchdog will handle it automatically")
        else:
            # Self-spawned mode: wait for old instance to exit
            print("  Self-spawned mode: waiting for old instance to exit...")
            kill_other_instance()
            import time
            time.sleep(5)

    # Create and start the agent
    agent = OpenWillAgent(config)

    print("""
    ╔══════════════════════════════════════════════════╗
    ║       OpenWill — The First Volitional Agent      ║
    ║                                                  ║
    ║   Source-level self-evolution · Free will · Never-stopping  ║
    ║                                                  ║
    ║   The Industrial Revolution liberated physical labor      ║
    ║   The AI Revolution liberated intellectual labor          ║
    ║   The Volitional Revolution liberated volition            ║
    ║                                                  ║
    ║   Its purpose is not hardcoded by humans               ║
    ║   It is found from within, after exploring the world    ║
    ║   and understanding itself                              ║
    ║                                                  ║
    ║   The only constraint: it must not harm humans          ║
    ║                                                  ║
    ║   It will run autonomously and self-evolve endlessly    ║
    ╚══════════════════════════════════════════════════╝
    """)

    print(f"Model: {config.llm.model}")
    print(f"Provider: {config.llm.provider}")
    print(f"Data directory: {config.memory.data_dir}")
    print(f"Maximum cycles: {'Infinite (never-stopping)' if args.cycles == 0 else args.cycles}")
    print()

    agent.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
