/**
 * The Learning lab's answers in plain words. A held-out question's right answer and the model's answers arrive as tool
 * calls, either as the eval's JSON ({"tool_calls": [{"name": "episodes_between", "arguments": {...}}]}) or as the model's
 * own markup (<tool_call><function=episodes_between><parameter=start>...</parameter></function></tool_call>). The screen
 * says what the call does ("looked back from 12:44 AM to 1:09 AM"); the raw text stays one press away.
 *
 * No imports at run time, so `node scripts/check-lab.mjs` can run this file as it is.
 */

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

/** The tool calls in an answer, or null when it holds none (a plain sentence). */
export function toolCallsOf(text: string | null | undefined): ToolCall[] | null {
  const raw = (text ?? "").trim();
  if (!raw) return null;
  if (raw.startsWith("{") || raw.startsWith("[")) {
    try {
      const parsed = JSON.parse(raw) as { tool_calls?: unknown } | unknown[];
      const list = Array.isArray(parsed) ? parsed : (parsed as { tool_calls?: unknown }).tool_calls;
      if (Array.isArray(list)) {
        const calls = list
          .map((c) => c as { name?: unknown; arguments?: unknown; function?: { name?: unknown; arguments?: unknown } })
          .map((c) => {
            const name = c.name ?? c.function?.name;
            let args = c.arguments ?? c.function?.arguments ?? {};
            if (typeof args === "string") {
              try {
                args = JSON.parse(args);
              } catch {
                args = {};
              }
            }
            return typeof name === "string" ? { name, arguments: (args ?? {}) as Record<string, unknown> } : null;
          })
          .filter((c): c is ToolCall => c !== null);
        return calls.length ? calls : null;
      }
    } catch {
      /* not JSON: try the markup */
    }
  }
  const calls: ToolCall[] = [];
  const fn = /<function=([^>\s]+)>([\s\S]*?)<\/function>/g;
  let m: RegExpExecArray | null;
  while ((m = fn.exec(raw)) !== null) {
    const args: Record<string, unknown> = {};
    const param = /<parameter=([^>\s]+)>([\s\S]*?)<\/parameter>/g;
    let p: RegExpExecArray | null;
    while ((p = param.exec(m[2] ?? "")) !== null) args[p[1]!] = (p[2] ?? "").trim();
    calls.push({ name: m[1]!, arguments: args });
  }
  return calls.length ? calls : null;
}

/** A time as the screen says it: the local clock time, with the date only when it is not `today`. */
export function clockWords(value: unknown, today: Date = new Date()): string {
  const s = String(value ?? "").trim();
  const d = new Date(s);
  if (!s || Number.isNaN(d.getTime())) return s || "an unstated time";
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const same = d.getFullYear() === today.getFullYear() && d.getMonth() === today.getMonth() && d.getDate() === today.getDate();
  return same ? time : `${d.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
}

function arg(call: ToolCall, ...names: string[]): unknown {
  for (const n of names) if (call.arguments[n] !== undefined && call.arguments[n] !== null && call.arguments[n] !== "") return call.arguments[n];
  return undefined;
}

/** One tool call as a verb phrase: "looked back from 12:44 AM to 1:09 AM", "checked where keys was last seen". */
export function callWords(call: ToolCall, today: Date = new Date()): string {
  switch (call.name) {
    case "episodes_between":
      return `looked back from ${clockWords(arg(call, "start", "from"), today)} to ${clockWords(arg(call, "end", "to"), today)}`;
    case "last_seen":
      return `checked where ${String(arg(call, "label", "object", "name") ?? "it")} was last seen`;
    case "search_memory":
      return `searched memory for "${String(arg(call, "q", "query", "text") ?? "")}"`;
    default:
      return call.name.replace(/[_-]+/g, " ").trim();
  }
}

/** An answer in plain words: its tool calls as verb phrases, or the sentence itself; "no answer" when empty. */
export function answerWords(text: string | null | undefined, today: Date = new Date()): string {
  const calls = toolCallsOf(text);
  if (calls) return calls.map((c) => callWords(c, today)).join(", then ");
  const plain = (text ?? "").replace(/\s+/g, " ").trim();
  return plain || "no answer";
}
