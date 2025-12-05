# benchcopy

Local copy prepared for two-way sync with [bechy53/bench](https://github.com/bechy53/bench).

## Remote configuration

This repository is configured with an `origin` remote pointing at `https://github.com/bechy53/bench` for both fetching and pushing.

You can verify the remote setup with:

```bash
git remote -v
```

## Syncing with GitHub

Use the following commands to pull the latest changes and push local commits once network access is available:

```bash
git fetch origin
git merge origin/main   # or the appropriate default branch once fetched

git push origin work    # replace `work` with the branch you want to push
```

> Note: If `git fetch origin` fails, check that outbound HTTPS access to GitHub is allowed in your environment. Once connectivity is available, the above commands will complete successfully.
