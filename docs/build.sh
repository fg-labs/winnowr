#!/usr/bin/env bash
# Builds the user guide (docs/) with mdBook and mdbook-linkcheck2, which fails the build on links
# to pages that do not exist. Read the Docs (.readthedocs.yaml) and CI (.github/workflows/docs.yml)
# both run it, so the tool versions and checksums live only here. The site is written to
# docs/book/html/.
#
# Usage: docs/build.sh
#
# By default the tools are the prebuilt x86_64 Linux release binaries, checked against their
# SHA-256 before they are unpacked. Elsewhere, set DOCS_TOOLS:
#   DOCS_TOOLS=path   use the mdbook and mdbook-linkcheck2 already on PATH
#   DOCS_TOOLS=cargo  build them with `cargo install` (into target/docs-tools)
#
# On Read the Docs, the site's path (e.g. /en/latest/) is taken from READTHEDOCS_CANONICAL_URL, so
# that the 404 page's links resolve.
set -euo pipefail

MDBOOK_VERSION=0.5.2
MDBOOK_SHA256=084e4342ba564db270108763e404a7d1f309d932651a22484e93c0dc1a071f6d
LINKCHECK2_VERSION=0.12.2
LINKCHECK2_SHA256=c68bba37dd07887d308a8c1ccab56aef8250793196e35c0fe517b9c7fbc44c48

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tools="$root/target/docs-tools"

# Downloads a release tarball, checks its SHA-256 and unpacks it into $tools/bin.
fetch() {
    local url=$1 sha256=$2 tarball
    tarball="$tools/$(basename "$url")"
    curl -fsSL --retry 3 --retry-connrefused "$url" -o "$tarball"
    echo "$sha256  $tarball" | sha256sum -c -
    tar -xzf "$tarball" -C "$tools/bin"
}

case "${DOCS_TOOLS:-prebuilt}" in
    prebuilt)
        if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
            echo "error: prebuilt tools are x86_64 Linux only; set DOCS_TOOLS=path or DOCS_TOOLS=cargo" >&2
            exit 1
        fi
        mkdir -p "$tools/bin"
        fetch "https://github.com/rust-lang/mdBook/releases/download/v$MDBOOK_VERSION/mdbook-v$MDBOOK_VERSION-x86_64-unknown-linux-gnu.tar.gz" \
            "$MDBOOK_SHA256"
        fetch "https://github.com/marxin/mdbook-linkcheck2/releases/download/v$LINKCHECK2_VERSION/mdbook-linkcheck2-x86_64-unknown-linux-gnu.tar.gz" \
            "$LINKCHECK2_SHA256"
        chmod +x "$tools/bin/mdbook" "$tools/bin/mdbook-linkcheck2"
        export PATH="$tools/bin:$PATH"
        ;;
    cargo)
        cargo install --locked --version "$MDBOOK_VERSION" --root "$tools" mdbook
        cargo install --locked --version "$LINKCHECK2_VERSION" --root "$tools" mdbook-linkcheck2
        export PATH="$tools/bin:$PATH"
        ;;
    path) ;;
    *)
        echo "error: DOCS_TOOLS must be prebuilt, cargo or path, got ${DOCS_TOOLS}" >&2
        exit 1
        ;;
esac

# Other versions may build the book differently from Read the Docs: say so, but build.
mdbook_version=$(mdbook --version)
linkcheck2_version=$(mdbook-linkcheck2 --version)
echo "$mdbook_version, $linkcheck2_version"
case "$mdbook_version $linkcheck2_version" in
    *"v$MDBOOK_VERSION "*"$LINKCHECK2_VERSION") ;;
    *) echo "warning: Read the Docs builds with mdbook $MDBOOK_VERSION and mdbook-linkcheck2 $LINKCHECK2_VERSION" >&2 ;;
esac

if [ -n "${READTHEDOCS_CANONICAL_URL:-}" ]; then
    # The URL's path, with a trailing slash: https://winnowr.readthedocs.io/en/latest/ -> /en/latest/
    site_path="/${READTHEDOCS_CANONICAL_URL#*://*/}"
    export MDBOOK_OUTPUT__HTML__SITE_URL="${site_path%/}/"
fi

mdbook build "$root/docs"
