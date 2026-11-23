
from src.database import ReferenceDatabase


def main():
    database = ReferenceDatabase()

    result = database.build("data/reference_faces")

    print("\nReference Database Results")
    print("--------------------------")
    print("Total photographs:", result["total_images"])
    print("Successfully processed:", result["successful"])
    print("Failed photographs:", len(result["errors"]))

    for error in result["errors"]:
        print("ERROR:", error)

    if result["successful"] > 0:
        saved_path = database.save()

        print("\nDatabase saved:", saved_path)
        print("Stored identities:")

        for name in database.identities:
            print("-", name)

        # Verify that the saved database can be loaded.
        another_database = ReferenceDatabase()
        count = another_database.load(saved_path)

        print("\nDatabase reload successful!")
        print("Loaded identities:", count)


if __name__ == "__main__":
    main()
