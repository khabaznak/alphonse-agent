/** Resolve a datetime-local value in the computer's timezone without guessing
 * across a daylight-saving gap or repeated (ambiguous) local time. */
export function resolveLocalReminderTime(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
  if (!match) throw new Error("invalid_local_time");
  const [, yearText, monthText, dayText, hourText, minuteText] = match;
  const year = Number(yearText), month = Number(monthText), day = Number(dayText);
  const hour = Number(hourText), minute = Number(minuteText);
  if (year < 100 || month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59) {
    throw new Error("invalid_local_time");
  }
  const matches = (date: Date) => date.getFullYear() === year && date.getMonth() + 1 === month
    && date.getDate() === day && date.getHours() === hour && date.getMinutes() === minute
    && date.getSeconds() === 0 && date.getMilliseconds() === 0;
  const local = new Date(year, month - 1, day, hour, minute, 0, 0);
  if (Number.isNaN(local.getTime()) || !matches(local)) throw new Error("nonexistent_local_time");
  // Timezone transitions may repeat a wall-clock minute. Search both sides
  // of the chosen instant, including date-line transitions, rather than
  // silently accepting JavaScript's arbitrary first occurrence.
  for (let delta = -26 * 60; delta <= 26 * 60; delta++) {
    if (delta !== 0 && matches(new Date(local.getTime() + delta * 60_000))) {
      throw new Error("ambiguous_local_time");
    }
  }
  return local.toISOString();
}
