//! The version of winnowr, with the git commit it was built from.

use std::sync::LazyLock;

/// The package version, followed by the short git commit hash it was built from and `-dirty`
/// when tracked files had uncommitted changes (see `build.rs`); the bare package version when
/// built outside a git checkout (e.g. from crates.io).
pub static VERSION: LazyLock<String> = LazyLock::new(|| {
    let dirty = option_env!("WINNOWR_GIT_DIRTY") == Some("true");
    version_string(env!("CARGO_PKG_VERSION"), option_env!("WINNOWR_GIT_HASH"), dirty)
});

/// [`VERSION`] followed by the versions of the statically linked libraries.
pub static LONG_VERSION: LazyLock<String> =
    LazyLock::new(|| format!("{}\nXGBoost {}", VERSION.as_str(), XGBOOST_VERSION.as_str()));

/// The version of the linked XGBoost library (e.g. `3.1.2`), as `--version` reports it and models
/// record it.
pub static XGBOOST_VERSION: LazyLock<String> = LazyLock::new(|| {
    let (major, minor, patch) = winnowr_xgboost::version();
    format!("{major}.{minor}.{patch}")
});

/// The package version of a [`VERSION`]-style string: `version` without its `-hash[-dirty]`
/// suffix, so the same release built from a git checkout and from a source tarball or crates.io
/// compare equal.
#[must_use]
pub fn package_version(version: &str) -> &str {
    let without_dirty = version.strip_suffix("-dirty").unwrap_or(version);
    match without_dirty.rsplit_once('-') {
        Some((package, hash))
            if !hash.is_empty() && hash.bytes().all(|b| b.is_ascii_hexdigit()) =>
        {
            package
        }
        _ => version,
    }
}

/// `package[-hash][-dirty]`.
fn version_string(package: &str, hash: Option<&str>, dirty: bool) -> String {
    let hash = hash.map(|hash| format!("-{hash}")).unwrap_or_default();
    let dirty = if hash.is_empty() || !dirty { "" } else { "-dirty" };
    format!("{package}{hash}{dirty}")
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::release("0.1.0", None, false, "0.1.0")]
    #[case::clean_checkout("0.1.0", Some("0123456789ab"), false, "0.1.0-0123456789ab")]
    #[case::dirty_checkout("0.1.0", Some("0123456789ab"), true, "0.1.0-0123456789ab-dirty")]
    #[case::dirty_without_hash("0.1.0", None, true, "0.1.0")]
    fn version_strings(
        #[case] package: &str,
        #[case] hash: Option<&str>,
        #[case] dirty: bool,
        #[case] expected: &str,
    ) {
        assert_eq!(version_string(package, hash, dirty), expected);
    }

    #[test]
    fn version_uses_this_build() {
        let hash = option_env!("WINNOWR_GIT_HASH");
        let dirty = option_env!("WINNOWR_GIT_DIRTY") == Some("true");
        assert_eq!(*VERSION, version_string(env!("CARGO_PKG_VERSION"), hash, dirty));
    }

    #[rstest]
    #[case::release("0.1.0", "0.1.0")]
    #[case::clean_checkout("0.1.0-0123456789ab", "0.1.0")]
    #[case::dirty_checkout("0.1.0-0123456789ab-dirty", "0.1.0")]
    #[case::pre_release("0.2.0-rc.1", "0.2.0-rc.1")]
    #[case::pre_release_checkout("0.2.0-rc.1-0123456789ab", "0.2.0-rc.1")]
    #[case::not_a_hash("0.1.0-beta", "0.1.0-beta")]
    fn package_versions(#[case] version: &str, #[case] expected: &str) {
        assert_eq!(package_version(version), expected);
    }

    #[test]
    fn long_version_reports_the_linked_xgboost() {
        let (major, minor, patch) = winnowr_xgboost::version();
        let expected = format!("{}\nXGBoost {major}.{minor}.{patch}", VERSION.as_str());
        assert_eq!(*LONG_VERSION, expected);
        assert!((major, minor) >= (3, 4), "{}", LONG_VERSION.as_str());
    }
}
