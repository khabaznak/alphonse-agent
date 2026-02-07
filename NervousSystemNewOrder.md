# Nervous System New Order

## Principle

Integrations are expressed through extremities. The extremities define inbound/outbound
behavior, and integrations are the concrete transport adapters they use.

## Current Order (Problem)

- Integrations live under `alphonse/extremities/interfaces/integrations/`.
- Extremities consume those integrations, but the structure implies integrations are
  primary and extremities are secondary.

## Proposed Order (Fix)

1) Extremities are the first-class layer.
2) Integrations are implementation details used by extremities.
3) Channel modules group both inbound and outbound extremities.

## Proposed Structure

```
alphonse/
  extremities/
    telegram/
      inbound.py        # (removed) legacy Telegram extremity
      outbound.py       # TelegramNotificationExtremity
      adapter.py        # TelegramAdapter
      config.py         # shared Telegram config
    webui/
      inbound.py        # Web UI input handler
      outbound.py       # Web UI output handler
  integration_adapters/
    telegram.py         # if we keep adapters isolated
```

## Notes

- This does not require immediate code movement.
- It establishes the direction: extremities first, adapters behind them.
