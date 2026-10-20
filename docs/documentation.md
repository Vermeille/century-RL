# Documentation and deployment

The site is built from `docs/` with MkDocs. `mkdocs.yml` defines the navigation
and strict link validation; the pinned standalone dependency lives in
`requirements-docs.txt`. Building docs does not import BoardRL, compile Cython,
install Torch, or require model files.

## Preview and build

From the repository root:

```bash
uvx --from mkdocs==1.6.1 mkdocs serve
```

Open <http://127.0.0.1:8000>. Stop the game server first, or add
`--dev-addr 127.0.0.1:8001` to the docs preview command. Build the static output:

```bash
uvx --from mkdocs==1.6.1 mkdocs build --strict
```

The output goes to ignored `site/`. Alternatively, create a separate Python
environment, install `pip install -r requirements-docs.txt`, and use
`mkdocs serve` / `mkdocs build --strict`. CI uses that requirements file.

MkDocs checks links between Markdown pages and their anchors during the build.
Use relative `.md` links for other guides; use full GitHub URLs for source files
outside `docs/`. The [MkDocs configuration reference](https://www.mkdocs.org/user-guide/configuration/)
documents the validation settings. This build does not check remote URL health
or prove that a shell example works.

## Automatic GitHub Pages publishing

`.github/workflows/docs.yml` builds every push to `master` and pull request to
`master`. Pull requests validate and upload the generated artifact. Successful
pushes to `master` deploy that artifact to the `github-pages` environment;
`workflow_dispatch` also permits a manual run from `master`.

For the first deployment, a repository administrator must enable **Settings →
Pages → Build and deployment → Source: GitHub Actions**. The workflow uses the
repository's `GITHUB_TOKEN` with Pages write and OIDC permissions in its deploy
job; no personal access token or `gh-pages` branch is needed. If environment
protection rules require deployment approval, those rules still apply. This
follows [GitHub's custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

For the current origin, the configured URL is
<https://vermeille.github.io/century-RL/>. The Actions deployment output is the
authoritative published URL. Adding the workflow locally does not publish it:
the changes must reach `master` and Pages must be enabled. On a fork, update
`site_url`, `repo_url`, and the edit link in `mkdocs.yml`.

This deploys the documentation, not a live game server. The
[workbench hosting guide](web-server.md#hosting) explains that separate service.

## Maintain executable examples

`tests/test_documentation.py` executes the marked Python blocks in the game
and custom-loss tutorials and runs the minimal training program through saving
and loading. Run:

```bash
OMP_NUM_THREADS=2 uv run pytest tests/test_documentation.py
```

These are short API smoke checks, not learning-performance tests. Keep partial
recipe fragments clearly described as excerpts with their required objects,
and reserve runnable labels for complete programs. Historical experiment notes
retain measured results with their protocol and caveats; do not turn them into
unverified claims about current defaults.
