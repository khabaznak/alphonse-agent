//! Navigation decisions for the protected, embedded owner Desktop.
//!
//! Only application assets may occupy its WebView. External web links are
//! opened by the OS instead; all other schemes are rejected.

use tauri::Url;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Destination {
    Application,
    ExternalWeb,
    Denied,
}

pub(crate) fn destination(url: &Url) -> Destination {
    if !url.username().is_empty() || url.password().is_some() {
        return Destination::Denied;
    }
    if url.scheme() == "tauri" && url.host_str() == Some("localhost") && url.port().is_none() {
        return Destination::Application;
    }
    if matches!(url.scheme(), "http" | "https") && url.host_str().is_some() {
        return Destination::ExternalWeb;
    }
    Destination::Denied
}

pub(crate) fn open_external(url: &Url) {
    if destination(url) == Destination::ExternalWeb {
        // No shell and no caller-chosen executable. A failed open does not
        // make it safe to load the destination inside the privileged WebView.
        let _ = std::process::Command::new("/usr/bin/open")
            .arg(url.as_str())
            .spawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_embedded_origin_loads_inside_webview() {
        for value in [
            "tauri://localhost/",
            "tauri://localhost/index.html#conversation",
        ] {
            assert_eq!(
                destination(&Url::parse(value).unwrap()),
                Destination::Application
            );
        }
        for value in ["https://example.com/a", "http://127.0.0.1:1420/"] {
            assert_eq!(
                destination(&Url::parse(value).unwrap()),
                Destination::ExternalWeb
            );
        }
        for value in [
            "tauri://other/index.html",
            "tauri://localhost:55/",
            "file:///tmp/a.html",
            "javascript:alert(1)",
            "https://user:pass@example.com/",
        ] {
            assert_eq!(
                destination(&Url::parse(value).unwrap()),
                Destination::Denied
            );
        }
    }
}
