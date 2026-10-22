# AI Cookbook

Sample code that accompanies blog posts about building with AI. Each folder is a self-contained example: a notebook or script you can run, plus its own README with setup steps and the packages it needs.

## Examples

| Folder | Blog post | What it shows |
| --- | --- | --- |
| [video-embedding](video-embedding/) | *Semantic Video Search for Media Archives* | Search a video by meaning, with text or a reference image, using Qwen3-VL embeddings, FAISS and an optional reranker. |

## Using an example

1. Open the folder and read its README. It lists the models, services and hardware the example expects.
2. Create a virtual environment and install that folder's `requirements.txt`.
3. Run the notebook top to bottom, or the script if the folder has one.

Examples don't share dependencies, so each folder gets its own environment.

## Repository layout

```
ai-cookbook/
├── images/            shared figures used by the READMEs and notebooks
└── <example>/         one folder per blog post
    ├── README.md
    ├── requirements.txt
    └── notebook and/or scripts
```

## Adding an example

- Create a new folder named after the topic, with its own `README.md` and `requirements.txt`.
- Put figures in the root `images/` folder and link them relatively (for example `../images/figure.png` from inside an example folder).
- Don't commit datasets, model weights or media you don't have the right to redistribute. Document where to get them instead, and add the download location to `.gitignore`.
- Add a row to the table above.

## Media and licenses

Media used in the examples keeps its own license; each example's README says where the files come from and what terms apply.
