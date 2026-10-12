# Third-party notices

Dependencies and downloaded model weights retain their own licenses. This repository does not bundle Laya model weights or third-party Python packages.

The referenced [Laya multilingual model](https://huggingface.co/convaiinnovations/laya-multilingual) identifies its license as Apache 2.0 in its model card. The downloader records the revision and artifact hashes; inspect the selected revision's terms when redistributing or adapting its weights. The plugin's software license does not replace upstream terms.

The bundled runtime evidence collection contains brief, attributed paraphrases and engineering inferences. Source articles, documentation, company names and trademarks remain with their respective owners. Third-party diagrams are not included. Company references imply no affiliation or endorsement.

Source URLs and available publication dates are recorded in `src/arch_review/data/sources.json` and the evidence records. All source summaries currently await independent expert review.

The separate research corpus in `datasets/open-reference-v1` contains licensed text from Donne Martin and contributors' **System Design Primer** (CC BY 4.0) and named chapter authors of **The Architecture of Open Source Applications** (CC BY 3.0). It is outside the MIT software grant. Full source attribution, pinned URLs, license links, and extraction changes are recorded per document and chunk. Upstream license notices are included in `datasets/open-reference-v1/licenses/`.

The scenarios and lessons in `datasets/source-linked-v1` are AI-authored adaptations with the corresponding source license and attribution recorded per item. They are hypothetical examples, not endorsed architecture assessments or human-reviewed labels. The research corpus is distributed in the repository/source archive; it is not automatically added to the lightweight runtime wheel's evidence index.
