from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

CONFIG_PATH = Path(__file__).with_name("job_book_config.json")


@dataclass
class FileRule:
    pattern: str
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "FileRule":
        return cls(pattern=data["pattern"], enabled=data.get("enabled", True))

    def to_dict(self) -> dict:
        return {"pattern": self.pattern, "enabled": self.enabled}


@dataclass
class JobBookConfig:
    search_root: str = "."
    file_rules: List[FileRule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "JobBookConfig":
        rules = [FileRule.from_dict(item) for item in data.get("file_rules", [])]
        return cls(search_root=data.get("search_root", "."), file_rules=rules)

    def to_dict(self) -> dict:
        return {
            "search_root": self.search_root,
            "file_rules": [rule.to_dict() for rule in self.file_rules],
        }

    def add_rule(self, pattern: str, enabled: bool = True) -> None:
        if not any(rule.pattern == pattern for rule in self.file_rules):
            self.file_rules.append(FileRule(pattern=pattern, enabled=enabled))

    def remove_rule(self, pattern: str) -> None:
        self.file_rules = [rule for rule in self.file_rules if rule.pattern != pattern]

    def set_rule_state(self, pattern: str, enabled: bool) -> None:
        for rule in self.file_rules:
            if rule.pattern == pattern:
                rule.enabled = enabled
                break
        else:
            self.file_rules.append(FileRule(pattern=pattern, enabled=enabled))

    def enabled_patterns(self) -> List[str]:
        return [rule.pattern for rule in self.file_rules if rule.enabled]


def load_config(path: Path = CONFIG_PATH) -> JobBookConfig:
    if not path.exists():
        return JobBookConfig()
    data = json.loads(path.read_text())
    return JobBookConfig.from_dict(data)


def save_config(config: JobBookConfig, path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(config.to_dict(), indent=2))


def scan_files(config: JobBookConfig) -> List[Path]:
    root = Path(config.search_root).expanduser().resolve()
    matches = []
    for pattern in config.enabled_patterns():
        matches.extend(root.rglob(pattern))
    # Deduplicate while preserving order
    seen = set()
    unique_matches: List[Path] = []
    for match in matches:
        resolved = match.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_matches.append(resolved)
    return unique_matches


def format_config(config: JobBookConfig) -> str:
    lines = [f"Search root: {Path(config.search_root).expanduser().resolve()}", "Patterns:"]
    if not config.file_rules:
        lines.append("  (none configured)")
    else:
        for rule in config.file_rules:
            state = "enabled" if rule.enabled else "disabled"
            lines.append(f"  - {rule.pattern} ({state})")
    return "\n".join(lines)


def configure_command(args: argparse.Namespace) -> None:
    config = load_config()

    if args.root:
        config.search_root = args.root

    for pattern in args.add or []:
        config.add_rule(pattern)

    for pattern in args.remove or []:
        config.remove_rule(pattern)

    for pattern in args.enable or []:
        config.set_rule_state(pattern, True)

    for pattern in args.disable or []:
        config.set_rule_state(pattern, False)

    save_config(config)

    if args.list_config:
        print(format_config(config))


def scan_command(args: argparse.Namespace) -> None:
    config = load_config()
    matches = scan_files(config)
    if not matches:
        print("No matching files found.")
        return
    print("\n".join(str(match) for match in matches))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Job Book Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure_parser = subparsers.add_parser("configure", help="Adjust file selection settings")
    configure_parser.add_argument("--root", help="Folder to search for files")
    configure_parser.add_argument("--add", nargs="*", help="Add file pattern(s) to track")
    configure_parser.add_argument("--remove", nargs="*", help="Remove file pattern(s) from tracking")
    configure_parser.add_argument("--enable", nargs="*", help="Enable pattern(s) for scanning")
    configure_parser.add_argument("--disable", nargs="*", help="Disable pattern(s) from scanning")
    configure_parser.add_argument(
        "--list-config",
        action="store_true",
        help="Display the saved configuration after updates",
    )
    configure_parser.set_defaults(func=configure_command)

    scan_parser = subparsers.add_parser("scan", help="Scan using the saved configuration")
    scan_parser.set_defaults(func=scan_command)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
