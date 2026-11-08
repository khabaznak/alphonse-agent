import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { AccountDisclosure } from './OptionalIntegrations';

describe('account processing disclosure', () => {
  it('explains external processing and local retention with an unchecked acknowledgement', () => {
    const html = renderToStaticMarkup(<AccountDisclosure accepted={false} onChange={() => {}} />);
    expect(html).toContain('email or calendar information');
    expect(html).toContain('TypeSafe and OpenAI');
    expect(html).toContain('local storage or backups');
    expect(html).toContain('I have reviewed how account information may be processed');
    expect(html).not.toContain('checked=""');
  });
});
