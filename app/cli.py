"""Quick manual test: python -m app.cli "your question" """
import logging
import sys

from app.config import get_settings
from app.rag.generation import answer_question


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    question = " ".join(sys.argv[1:]) or "Which extensions are mandatory in RVA23?"
    s = get_settings()
    print(f"[config] chat_model={s.chat_model} db_path={s.db_path} final_k={s.final_k}\n")
    answer, chunks = answer_question(question)
    print(answer)
    print("\nSources:")
    for c in chunks:
        print(f"  - {c.metadata.get('source')} pages {c.metadata.get('pages')}")


if __name__ == "__main__":
    main()
