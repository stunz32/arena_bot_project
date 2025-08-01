import os
from pathlib import Path
from datetime import datetime
import traceback

# --- THIS IS THE EXACT PATH THE BOT TRIES TO USE ---
# We are testing the Windows path directly.
LOGS_BASE_PATH_TO_TEST = "M:\\Hearthstone\\Logs"

def diagnose_path_logic():
    """
    Performs a deep diagnostic on the log directory scanning logic.
    """
    print("🎯 DEEP PATH LOGIC DIAGNOSTIC")
    print("=" * 60)
    print(f"Python version: {os.sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print("-" * 60)

    base_path = Path(LOGS_BASE_PATH_TO_TEST)

    # 1. Check base path existence and type
    print(f"1. Checking base path: '{base_path}'")
    if not base_path.exists():
        print(f"   ❌ CRITICAL FAILURE: The base path does not exist according to Python.")
        return
    print(f"   ✅ Path exists.")

    if not base_path.is_dir():
        print(f"   ❌ CRITICAL FAILURE: The path is not a directory.")
        return
    print(f"   ✅ Path is a directory.")

    # 2. Try to iterate through the directory contents
    print("\n2. Iterating through directory contents...")
    try:
        items = list(base_path.iterdir())
        print(f"   ✅ Found {len(items)} items in the directory.")
    except Exception as e:
        print(f"   ❌ CRITICAL FAILURE: Could not list items in the directory.")
        print(f"      Error: {e}")
        traceback.print_exc()
        return

    # 3. Analyze each item, replicating the bot's logic
    print("\n3. Analyzing each item...")
    found_directories = []
    for item in items:
        print(f"\n   --- Analyzing item: '{item.name}' ---")
        try:
            # Check if it's a directory
            is_directory = item.is_dir()
            print(f"   Is it a directory? -> {is_directory}")
            if not is_directory:
                print("      SKIPPING (Not a directory).")
                continue

            # Check if the name starts correctly
            name_starts_correctly = item.name.startswith("Hearthstone_")
            print(f"   Name starts with 'Hearthstone_'? -> {name_starts_correctly}")
            if not name_starts_correctly:
                print("      SKIPPING (Name format incorrect).")
                continue
            
            # Try to parse the timestamp (the most likely failure point)
            print(f"   Attempting to parse timestamp from '{item.name}'...")
            try:
                parts = item.name.split("_")
                print(f"      Split parts: {parts}")
                if len(parts) >= 7:
                    year, month, day, hour, minute, second = parts[1:7]
                    print(f"         Year: {year}, Month: {month}, Day: {day}")
                    print(f"         Hour: {hour}, Minute: {minute}, Second: {second}")
                    
                    timestamp = datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), int(second)
                    )
                    print(f"      ✅ Timestamp parsed successfully: {timestamp}")
                    found_directories.append((timestamp, item))
                else:
                    print(f"      ❌ FAILED: Not enough parts after splitting by '_'. Expected >= 7, got {len(parts)}.")

            except Exception as parse_error:
                print(f"      ❌ FAILED: Error during timestamp parsing.")
                print(f"         Error details: {parse_error}")
                traceback.print_exc()

        except Exception as item_error:
            print(f"   ❌ FAILED: An unexpected error occurred while processing this item.")
            print(f"      Error details: {item_error}")
            traceback.print_exc()

    # 4. Final Result
    print("\n" + "=" * 60)
    print("4. DIAGNOSTIC COMPLETE")
    print(f"   ✅ Found {len(found_directories)} valid Hearthstone session directories.")

    if found_directories:
        found_directories.sort(key=lambda x: x[0], reverse=True)
        most_recent_timestamp, most_recent_path = found_directories[0]
        print(f"   🚀 Most recent directory identified:")
        print(f"      Path: {most_recent_path}")
        print(f"      Timestamp: {most_recent_timestamp}")
    else:
        print(f"   ❌ FAILURE: No valid session directories were found.")
        print(f"      This is why the bot returns 'None' and fails.")

if __name__ == "__main__":
    diagnose_path_logic()