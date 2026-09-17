"""Static TeX source audit; this is not a LaTeX compiler or layout check."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "manuscript"
SOURCES = [ROOT / "main.tex", ROOT / "supplementary/supplement.tex"]


def strip_comments(source: str) -> str:
    lines = []
    for line in source.splitlines():
        for i, char in enumerate(line):
            if char == "%":
                slashes = len(line[:i]) - len(line[:i].rstrip("\\"))
                if slashes % 2 == 0:
                    line = line[:i]
                    break
        lines.append(line)
    return "\n".join(lines)


def check_source(path: Path) -> list[str]:
    issues = []
    source = strip_comments(path.read_text(encoding="utf-8"))
    source = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", source, flags=re.S)
    if not source.count(r"\documentclass") == 1 or not source.rstrip().endswith(r"\end{document}"):
        issues.append("expected one documentclass and a final end{document}")
    stack = []
    for line_number, line in enumerate(source.splitlines(), 1):
        for token in re.finditer(r"(?<!\\)\\(begin|end)\{([^{}]+)\}", line):
            kind, name = token.groups()
            if kind == "begin":
                stack.append((name, line_number))
            elif not stack or stack[-1][0] != name:
                issues.append(f"line {line_number}: unmatched end{{{name}}}")
            else:
                stack.pop()
    issues.extend(f"line {line}: unclosed begin{{{name}}}" for name, line in stack)
    depth = 0
    for line_number, line in enumerate(source.splitlines(), 1):
        for i, char in enumerate(line):
            if char not in "{}":
                continue
            slashes = len(line[:i]) - len(line[:i].rstrip("\\"))
            if slashes % 2:
                continue
            depth += 1 if char == "{" else -1
            if depth < 0:
                issues.append(f"line {line_number}: closing brace without opener")
                depth = 0
    if depth:
        issues.append(f"{depth} unclosed braces")
    if len(re.findall(r"(?<!\\)(?<!\$)\$(?!\$)", source)) % 2:
        issues.append("odd number of inline math delimiters")
    labels = set(re.findall(r"\\label\{([^{}]+)\}", source))
    for label in re.findall(r"\\(?:ref|eqref)\{([^{}]+)\}", source):
        if label not in labels:
            issues.append(f"undefined local reference {label}")
    if path.name == "main.tex":
        bibkeys = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^{}]+)\}", source))
        for group in re.findall(r"\\cite\{([^{}]+)\}", source):
            for key in group.split(","):
                if key.strip() not in bibkeys:
                    issues.append(f"undefined citation {key.strip()}")
    for figure in re.findall(r"\\(?:safegraphic(?:\[[^]]*\])?|includegraphics(?:\[[^]]*\])?)\{([^{}]+)\}", source):
        if figure.startswith("#"):
            continue  # Macro definition argument, not an asset path.
        if not (path.parent / figure).is_file():
            issues.append(f"missing figure {figure}")
    return issues


def main() -> None:
    failures = 0
    for path in SOURCES:
        issues = check_source(path)
        print(f"{path.relative_to(ROOT)}: {'PASS' if not issues else 'FAIL'}")
        for issue in issues:
            print(f"  {issue}")
        failures += len(issues)
    if failures:
        raise SystemExit(1)
    print("Static syntax/assets/citations/references audit passed; compilation not verified.")


if __name__ == "__main__":
    main()
