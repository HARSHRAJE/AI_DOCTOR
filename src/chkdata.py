from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"

classes = ["COVID", "NORMAL", "PNEUMONIA"]

print("Class distribution")
print("=" * 30)

total = 0

for class_name in classes:
    class_dir = DATA_DIR / class_name

    count = len([
        p for p in class_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    print(f"{class_name:10} : {count}")
    total += count

print("=" * 30)
print(f"{'TOTAL':10} : {total}")