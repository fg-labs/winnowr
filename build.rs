//! Records the git commit winnowr is built from, for `winnowr --version`.
//!
//! Sets `WINNOWR_GIT_HASH` (a short commit hash) and `WINNOWR_GIT_DIRTY` (`true` when tracked
//! files have uncommitted changes) only when the package directory is the top of a git checkout:
//! a crate unpacked from crates.io, or vendored inside another repository, gets neither.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));

    // Printing any rerun-if-changed turns off Cargo's default of rerunning on every package
    // file change, so list the files whose edits change the dirty state, as well as the git
    // files that change on a commit, a checkout or staging.
    let code = [
        "build.rs",
        "Cargo.toml",
        "Cargo.lock",
        "src",
        "xgboost/build.rs",
        "xgboost/build",
        "xgboost/Cargo.toml",
        "xgboost/patches",
        "xgboost/src",
    ];
    for path in code {
        println!("cargo:rerun-if-changed={path}");
    }

    let Some(git) = GitCheckout::at(&manifest_dir) else {
        return;
    };
    for path in git.state_files() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    if let Some(hash) = git.output(&["rev-parse", "--short=12", "HEAD"]) {
        println!("cargo:rustc-env=WINNOWR_GIT_HASH={hash}");
        let dirty = git.output(&["status", "--porcelain", "--untracked-files=no"]);
        let dirty = dirty.is_some_and(|changes| !changes.is_empty());
        println!("cargo:rustc-env=WINNOWR_GIT_DIRTY={dirty}");
    }
}

/// A git checkout whose top level is the package directory.
struct GitCheckout {
    /// The checkout's top-level directory.
    top: PathBuf,
}

impl GitCheckout {
    /// The checkout at `dir`, if `dir` is the top of one (and git is available).
    fn at(dir: &Path) -> Option<Self> {
        let checkout = Self { top: dir.to_path_buf() };
        let top = PathBuf::from(checkout.output(&["rev-parse", "--show-toplevel"])?);
        let same = top.canonicalize().ok()? == dir.canonicalize().ok()?;
        same.then_some(checkout)
    }

    /// The trimmed standard output of a successful `git` command run in the checkout.
    fn output(&self, args: &[&str]) -> Option<String> {
        let output = Command::new("git").arg("-C").arg(&self.top).args(args).output().ok()?;
        output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// The git files that change when HEAD moves or the index changes: HEAD, the index, the
    /// current branch's ref and the packed refs (which live in the common directory of a
    /// worktree).
    fn state_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        if let Some(git_dir) = self.output(&["rev-parse", "--absolute-git-dir"]) {
            let git_dir = PathBuf::from(git_dir);
            files.extend([git_dir.join("HEAD"), git_dir.join("index")]);
        }
        let common = self.output(&["rev-parse", "--git-common-dir"]).map(|dir| self.top.join(dir));
        if let Some(common) = common {
            files.push(common.join("packed-refs"));
            if let Some(branch) = self.output(&["symbolic-ref", "-q", "HEAD"]) {
                files.push(common.join(branch));
            }
        }
        files.retain(|path| path.exists());
        files
    }
}
