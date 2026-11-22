import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { OwnerConversationRecoveryNotice } from "./OwnerConversationRecoveryNotice";

describe("conversation acknowledgement recovery", () => {
  it("renders a retry control beside the preserved composer draft", () => {
    const html = renderToStaticMarkup(<section>
      <textarea aria-label="Message Alphonse" value="Please add apples" readOnly />
      <OwnerConversationRecoveryNotice state="acknowledgement" retrying={false} onRetry={() => {}} />
    </section>);
    expect(html).toContain("Please add apples");
    expect(html).toContain("The answer is shown");
    expect(html).toContain("Retry saving answer");
    expect(html).toContain('type="button"');
    expect(html).not.toContain("disabled");
  });

  it("disables retry while in flight and offers no retry for uncertain pickup", () => {
    const retrying = renderToStaticMarkup(<OwnerConversationRecoveryNotice state="acknowledgement" retrying onRetry={() => {}} />);
    expect(retrying).toContain("disabled");
    const uncertain = renderToStaticMarkup(<OwnerConversationRecoveryNotice state="uncertain" retrying={false} onRetry={() => {}} />);
    expect(uncertain).toContain("uncertain result");
    expect(uncertain).not.toContain("Retry saving answer");
  });

  it("connects the retry control to its explicit handler", () => {
    const onRetry = vi.fn();
    const notice = OwnerConversationRecoveryNotice({ state: "acknowledgement", retrying: false, onRetry });
    const children = notice?.props.children as Array<unknown>;
    const button = children.find((child): child is { props: { onClick: () => void } } =>
      typeof child === "object" && child !== null && "props" in child);
    button?.props.onClick();
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
