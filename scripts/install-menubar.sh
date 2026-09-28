#!/usr/bin/env bash
# Install the Bsafe menu bar app: build Swift targets, assemble
# ~/Applications/Bsafe.app, ad-hoc sign it, and open it.
set -euo pipefail

NO_OPEN=0
for arg in "$@"; do
    case "$arg" in
        --no-open) NO_OPEN=1 ;;
        -h|--help)
            echo "Usage: $0 [--no-open]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$REPO/.venv/bin/bsafe"

if [[ ! -x "$CLI" ]]; then
    echo "Error: bsafe CLI not found at $CLI" >&2
    echo "Run 'uv sync' in $REPO first, then retry." >&2
    exit 1
fi

(cd "$REPO/swift" && swift build -c release)

BIN="$REPO/swift/.build/release/BsafeMenuBar"
if [[ ! -f "$BIN" ]]; then
    echo "Error: $BIN was not built." >&2
    exit 1
fi

APP_DIR="$HOME/Applications"
APP="$APP_DIR/Bsafe.app"
mkdir -p "$APP_DIR"

# Quit a running instance gracefully so willTerminate reaps CLI/helper.
# (Ignore errors when it is not running.)
osascript -e 'quit app "Bsafe"' >/dev/null 2>&1 || true
# Wait up to 5 s for the app to exit.
for i in $(seq 1 50); do
    if ! pgrep -x Bsafe >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
# Fallback: force-quit any remaining app instance.
if pgrep -x Bsafe >/dev/null 2>&1; then
    pkill -x Bsafe >/dev/null 2>&1 || true
    sleep 1
fi
# Reap any orphaned CLI whose parent is init (ppid 1). Match the full
# command line and require ppid 1 so an unrelated reused pid is never
# killed. The bracket regex avoids matching pgrep itself.
if command -v pgrep >/dev/null 2>&1; then
    for pid in $(pgrep -f "[.]venv/bin/bsafe start" 2>/dev/null || true); do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        if [ "$ppid" = "1" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in $(pgrep -f "[.]venv/bin/bsafe start" 2>/dev/null || true); do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        if [ "$ppid" = "1" ]; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$BIN" "$APP/Contents/MacOS/Bsafe"
chmod +x "$APP/Contents/MacOS/Bsafe"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.bsafe.menubar</string>
    <key>CFBundleName</key>
    <string>Bsafe</string>
    <key>CFBundleExecutable</key>
    <string>Bsafe</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>BsafeCLIPath</key>
    <string>$CLI</string>
</dict>
</plist>
EOF

# Sign with a stable identity when one exists so macOS privacy grants (Documents,
# Screen Recording) survive rebuilds; ad-hoc signatures change on every build.
SIGN_ID="${BSAFE_SIGN_IDENTITY:-}"
if [ -z "$SIGN_ID" ]; then
    SIGN_ID="$(security find-identity -v -p codesigning 2>/dev/null \
        | grep -E '"(Developer ID Application|Apple Development):' | head -1 \
        | sed -E 's/.*"(.*)"/\1/')" || true
fi
if [ -n "$SIGN_ID" ]; then
    echo "Signing with: $SIGN_ID"
    codesign --force --sign "$SIGN_ID" "$APP"
else
    echo "No signing identity found; signing ad-hoc (privacy grants reset on reinstall)."
    codesign --force --sign - "$APP"
fi

echo "Installed $APP"

if [[ "$NO_OPEN" -eq 0 ]]; then
    open "$APP"
fi

cat <<EOF
Note: on first Start, grant Screen Recording permission to Bsafe
(System Settings -> Privacy & Security -> Screen Recording).
The overlay helper is launched by the CLI, but the permission prompt
appears for Bsafe. With a signing identity the grant survives reinstalls;
ad-hoc signing (no identity) requires re-granting after each reinstall.
EOF
