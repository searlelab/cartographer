import sys


MIGRATION_MESSAGE = (
    'chronologer_trainer.py has been disabled.\n'
    'Use the UNIMOD parquet trainer instead:\n'
    '  python src/chronologer_unimod_trainer.py --dataset_root /path/to/prospect-ptms-irt'
)


def main():
    raise SystemExit(MIGRATION_MESSAGE)


if __name__ == '__main__':
    main()
    sys.exit()

