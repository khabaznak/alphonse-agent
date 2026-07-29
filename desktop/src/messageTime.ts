export type MessageTimeLabels = {
  visible: string;
  tooltip: string;
};

export function formatMessageTime(
  createdAt: string,
  timezone: string,
  now = new Date(),
): MessageTimeLabels | null {
  const instant = new Date(createdAt);
  if (!createdAt || Number.isNaN(instant.getTime())) return null;
  const zone = timezone.trim() || "UTC";
  try {
    const day = new Intl.DateTimeFormat("en-CA", {
      timeZone: zone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
    const isToday = day.format(instant) === day.format(now);
    const visible = new Intl.DateTimeFormat(undefined, {
      timeZone: zone,
      ...(isToday ? {} : { year: "numeric", month: "short", day: "numeric" }),
      hour: "numeric",
      minute: "2-digit",
    }).format(instant);
    const tooltip = new Intl.DateTimeFormat(undefined, {
      timeZone: zone,
      dateStyle: "full",
      timeStyle: "long",
    }).format(instant);
    return { visible, tooltip: `${tooltip} · ${zone} · ${instant.toISOString()}` };
  } catch (error) {
    if (error instanceof RangeError && zone !== "UTC") return formatMessageTime(createdAt, "UTC", now);
    return null;
  }
}
