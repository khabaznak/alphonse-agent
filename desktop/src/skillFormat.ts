// Match Python and JSON Schema character limits, including supplementary Unicode.
export const SKILL_NAME_MAX_LENGTH = 64;
export const SKILL_DESCRIPTION_MAX_LENGTH = 1024;
export const skillTextLength = (value: string) => Array.from(value).length;
export const limitSkillText = (value: string, maximum: number) => Array.from(value).slice(0, maximum).join("");
