"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root  # (1)!

for path in sorted(src.rglob("*.py")):  # (2)!
    module_path = path.relative_to(src).with_suffix("")  # (3)!
    doc_path = path.relative_to(src).with_suffix(".md")  # (4)!
    full_doc_path = Path("reference", doc_path)  # (5)!

    parts = tuple(module_path.parts)

    if parts[0] != "heat":
        continue

    if parts[-1] == "__init__":  # (6)!
        continue
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    if "tests" in parts:
        continue

    nav[parts] = doc_path.as_posix()  # (1)!

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  # (7)!
        identifier = ".".join(parts)  # (8)!
        print("::: " + identifier, file=fd)  # (9)!

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))  # (10)!

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  # (2)!
    nav_file.writelines(nav.build_literate_nav())  # (3)!
