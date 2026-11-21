"""Host-declared prerequisites that are known to have made no effect."""


class MissingPrerequisite(Exception):
    """A fenced operation refused before making any effect; recovery is safe."""

    def __init__(self, code):
        if code not in {'owner_clock', 'owner_schedules', 'owner_schedule_time'}:
            raise ValueError('unknown_prerequisite')
        self.code = code
        super().__init__(code)


FEEDBACK = {
    'owner_clock': 'The previous action was not executed because a current owner clock observation is required. '
                   'Use the available clock tool, then reconsider the request and the action.',
    'owner_schedules': 'The previous action was not executed because a current schedule list is required. '
                       'Use owner_schedules to identify the intended task; ask if it is ambiguous.',
    'owner_schedule_time': 'The previous action was not executed because its scheduled time was invalid or not in the future. '
                           'Use the clock and supply an actual ISO datetime with an explicit offset, or ask if the requested time is unclear.',
}
