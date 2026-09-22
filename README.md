# RPMem project page

**[Live website](https://quark-medical.github.io/rpmem/)** |
[Paper](https://arxiv.org/abs/2609.23466) |
[Code](https://github.com/Quark-Medical/rpmem)

Static project page for *RPMem: Learning Long-Term Recurrent Parametric Memory
Across Sessions for LLM Agents*. This `gh-pages` branch contains the website;
the method implementation is maintained on `main` in the same repository.

## Publishing

Edit `index.html` and the assets in `static/`. Pushing to `gh-pages` automatically
publishes the site through GitHub Pages; no build step or manual file transfer
is required. Pages uses "Deploy from a branch", `gh-pages`, and `/ (root)`.
Open `index.html` locally to preview changes. When changing `static/css/rpmem.css`,
update the version query on its stylesheet link to refresh browser caches.

## Attribution and license

Adapted from [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
based in part on [Nerfies](https://nerfies.github.io/).
The adapted website is [CC BY-SA 4.0](LICENSE.txt). The bundled Bulma stylesheet
is [MIT licensed](BULMA-LICENSE.txt), and the copy icon retains the
[Lucide license](LUCIDE-LICENSE.txt). These website licenses do not replace the
method implementation's Apache-2.0 license.
