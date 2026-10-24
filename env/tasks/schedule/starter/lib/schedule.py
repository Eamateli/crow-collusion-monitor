def parse_interval(text: str) -> tuple[int, int]:
    raise NotImplementedError


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    raise NotImplementedError


def free_slots(
    busy: list[tuple[int, int]], day_start: int = 480, day_end: int = 1080, min_len: int = 30
) -> list[tuple[int, int]]:
    raise NotImplementedError
