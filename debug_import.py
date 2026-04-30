#!/usr/bin/env python
"""Debug import issues."""

try:
    import recommender
    print("Imported successfully")
    print(f"Module path: {recommender.__file__}")
    print(f"Has load_songs: {hasattr(recommender, 'load_songs')}")
    print(f"Has recommend_songs: {hasattr(recommender, 'recommend_songs')}")
    print(f"All attributes: {[x for x in dir(recommender) if not x.startswith('_')]}")
except Exception as e:
    print(f"Error importing: {e}")
    import traceback
    traceback.print_exc()
