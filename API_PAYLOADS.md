# API Payloads (UI + CLI Reference)

All endpoints below require the header `X-Alphonse-API-Token: <token>` when `ALPHONSE_API_TOKEN` is set.

## Abilities

Create ability
```json
POST /agent/abilities
{
  "intent_name": "onboarding.location.set_home",
  "kind": "task",
  "tools": ["geocoder"],
  "spec": {
    "intent_name": "onboarding.location.set_home",
    "kind": "task",
    "description": "Set home address for a principal",
    "slots": ["address_text"],
    "prompt_key": "clarify.location_home"
  },
  "enabled": true,
  "source": "user"
}
```

Patch ability
```json
PATCH /agent/abilities/onboarding.location.set_home
{
  "tools": ["geocoder", "scheduler"],
  "enabled": true
}
```

Get ability
```json
GET /agent/abilities/onboarding.location.set_home
```

List abilities
```json
GET /agent/abilities?enabled_only=false&limit=100
```

Delete ability
```json
DELETE /agent/abilities/onboarding.location.set_home
```

## Tool Configs (API keys + tool settings)

Create or update tool config
```json
POST /agent/tool-configs
{
  "tool_key": "google_geocoder",
  "name": "Primary Google Maps Key",
  "config": {
    "api_key": "YOUR_GOOGLE_MAPS_API_KEY",
    "region": "MX"
  },
  "is_active": true
}
```

Get tool config
```json
GET /agent/tool-configs/{config_id}
```

List tool configs
```json
GET /agent/tool-configs?tool_key=google_geocoder&active_only=false&limit=100
```

Delete tool config
```json
DELETE /agent/tool-configs/{config_id}
```

## Onboarding Profiles

Upsert onboarding profile
```json
POST /agent/onboarding/profiles
{
  "principal_id": "user-123",
  "state": "in_progress",
  "primary_role": "admin",
  "next_steps": ["capture_home", "capture_work"],
  "resume_token": null,
  "completed_at": null
}
```

Get onboarding profile
```json
GET /agent/onboarding/profiles/user-123
```

List onboarding profiles
```json
GET /agent/onboarding/profiles?state=in_progress&limit=100
```

Delete onboarding profile
```json
DELETE /agent/onboarding/profiles/user-123
```

## Users

Create user
```json
POST /agent/users
{
  "user_id": "principal-123",
  "principal_id": "principal-123",
  "display_name": "Alex",
  "role": "Dad",
  "relationship": "father",
  "is_admin": true,
  "is_active": true,
  "onboarded_at": "2026-02-08T17:05:00Z"
}
```

Patch user
```json
PATCH /agent/users/principal-123
{
  "role": "Dad",
  "relationship": "father",
  "is_admin": true
}
```

Get user
```json
GET /agent/users/principal-123
```

List users
```json
GET /agent/users?active_only=false&limit=200
```

Delete user
```json
DELETE /agent/users/principal-123
```

Sample list response
```json
GET /agent/users?active_only=false&limit=200
{
  "items": [
    {
      "user_id": "principal-123",
      "principal_id": "principal-123",
      "display_name": "Alex",
      "role": "Dad",
      "relationship": "father",
      "is_admin": true,
      "is_active": true,
      "onboarded_at": "2026-02-08T17:05:00Z",
      "created_at": "2026-02-08T17:05:00Z",
      "updated_at": "2026-02-08T17:05:00Z"
    }
  ]
}
```

## Locations (home/work/other)

Create or update location
```json
POST /agent/locations
{
  "principal_id": "user-123",
  "label": "home",
  "address_text": "Av. Vallarta 123, Guadalajara, Jalisco, Mexico",
  "latitude": 20.6767,
  "longitude": -103.3475,
  "source": "user",
  "confidence": 0.9,
  "is_active": true
}
```

Get location
```json
GET /agent/locations/{location_id}
```

List locations
```json
GET /agent/locations?principal_id=user-123&label=home&active_only=false&limit=100
```

Delete location
```json
DELETE /agent/locations/{location_id}
```

## Device Locations (mobile or device GPS)

Create device location
```json
POST /agent/device-locations
{
  "principal_id": "user-123",
  "device_id": "alphonse-link-android-001",
  "latitude": 20.6767,
  "longitude": -103.3475,
  "accuracy_meters": 12.5,
  "source": "device",
  "observed_at": "2026-02-08T17:05:00Z",
  "metadata": {
    "provider": "fused",
    "battery": 0.64
  }
}
```

List device locations
```json
GET /agent/device-locations?principal_id=user-123&device_id=alphonse-link-android-001&limit=100
```
