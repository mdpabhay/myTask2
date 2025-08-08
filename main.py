# main.py
import os
import re
import io
import math
import glob
import shutil
import hashlib
import tempfile
import logging
from typing import List, Tuple, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from datasketch import MinHash

# Optional heavy dependencies:
try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except Exception:
    GITHUB_AVAILABLE = False

try:
    import mosspy
    MOSS_AVAILABLE = True
except Exception:
    MOSS_AVAILABLE = False

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except Exception:
    DATASKETCH_AVAILABLE = False

# Optional semantic embeddings (heavy). If unavailable, semantic step is skipped.
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# CONFIG
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
MOSS_USER_ID = os.getenv("MOSS_USER_ID", None)  # optional
# Winnowing params
SHINGLE_K = 5     # number of tokens per shingle
WINNOW_W = 4      # winnowing window size
MINHASH_PERM = 128
LSH_THRESHOLD = 0.2  # candidate retrieval threshold
LSH_BANDS = 16

# Language extension mapping (extendable)
LANG_EXTENSIONS = {
    "python": [".py"],
    "java": [".java"],
    "javascript": [".js", ".jsx", ".ts", ".tsx"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
    "csharp": [".cs"],
    "go": [".go"],
    "ruby": [".rb"],
    "php": [".php"],
    "rust": [".rs"],
    "kotlin": [".kt", ".kts"],
    "shell": [".sh"],
    "perl": [".pl"],
    "r": [".r"],
    "matlab": [".m"],
    "html": [".html", ".htm"],
    "css": [".css"],
    # add more as needed
}

# Utility helpers -----------------------------------------------------------
def is_text_file(path: str) -> bool:
    # Skip obvious binary files by extension
    binary_exts = {'.png', '.jpg', '.jpeg', '.gif', '.class', '.exe', '.dll', '.so', '.o', '.pdf'}
    ext = os.path.splitext(path)[1].lower()
    if ext in binary_exts:
        return False
    # small heuristic: try opening
    try:
        with open(path, 'rb') as f:
            chunk = f.read(4096)
            if b'\0' in chunk:
                return False
        return True
    except Exception:
        return False

def is_vendor_dir(path: str) -> bool:
    vendor_names = {'node_modules', 'venv', '.venv', '.git', 'dist', 'build', '__pycache__', 'vendor'}
    return any(part in vendor_names for part in path.split(os.sep))

def detect_files_by_language(repo_root: str) -> Dict[str, List[str]]:
    files_by_lang = {}
    for root, dirs, files in os.walk(repo_root):
        # skip vendor directories entirely
        if is_vendor_dir(root):
            continue
        for fname in files:
            fpath = os.path.join(root, fname)
            if not is_text_file(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            for lang, exts in LANG_EXTENSIONS.items():
                if ext in exts:
                    files_by_lang.setdefault(lang, []).append(fpath)
                    break
    return files_by_lang

# Normalization (best-effort comment removal + literal masking)
def remove_comments_and_strings(text: str, lang: str) -> str:
    # Best-effort patterns for many languages. Not perfect — language-specific parsers (tree-sitter) are recommended.
    # Remove triple-quoted strings for Python
    if lang == 'python':
        text = re.sub(r'("""|\'\'\')(?:.|[\r\n])*?\1', ' <STR> ', text, flags=re.MULTILINE)
    # Remove common string literals
    text = re.sub(r'("([^"\\]|\\.)*")|(\'([^\'\\]|\\.)*\')', ' <STR> ', text)
    # Remove single-line comments
    text = re.sub(r'//.*', ' ', text)   # C/Java/JS style
    text = re.sub(r'#.*', ' ', text)    # Python / shell
    # Remove multi-line comments /* ... */
    text = re.sub(r'/\*(?:.|[\r\n])*?\*/', ' ', text)
    # Remove docstrings left
    text = re.sub(r'""".*?"""', ' ', text, flags=re.S)
    return text

def mask_literals(text: str) -> str:
    # mask numbers
    text = re.sub(r'\b\d+(\.\d+)?\b', ' <NUM> ', text)
    # mask long strings already replaced; small strings covered above
    return text

def tokenize_text(text: str) -> List[str]:
    # split on non-word characters, keep identifiers, keywords
    tokens = re.findall(r'[A-Za-z_]\w*|==|!=|<=|>=|->|::|<=|>=|[{}()\[\];.,:+\-*/%&|^~<>]', text)
    return tokens

# Shingle & winnowing
def shingle_tokens(tokens: List[str], k: int = SHINGLE_K) -> List[str]:
    if len(tokens) < k:
        return [' '.join(tokens)]
    return [' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1)]

def hash_shingle(sh: str) -> int:
    # deterministic 64-bit hash
    h = hashlib.sha256(sh.encode('utf-8')).hexdigest()
    return int(h[:16], 16)

def winnow_hashes(hashes: List[int], w: int = WINNOW_W) -> set:
    if not hashes:
        return set()
    fingerprints = set()
    for i in range(0, max(1, len(hashes)-w+1)):
        window = hashes[i:i+w]
        min_hash = min(window)
        fingerprints.add(min_hash)
    # If not enough hashes to form a window, include all
    if len(hashes) < w:
        fingerprints.update(hashes)
    return fingerprints

# MinHash wrapper
def build_minhash_from_fingerprints(fp_set: set, num_perm: int = MINHASH_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for fp in fp_set:
        m.update(str(fp).encode('utf8'))
    return m

# Semantic embeddings (optional)
EMBED_MODEL = None
def init_embedding_model():
    global EMBED_MODEL
    if SBERT_AVAILABLE and EMBED_MODEL is None:
        try:
            EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast; replace with Code-specific model if desired
        except Exception as e:
            logging.warning("SentenceTransformer init failed: %s", e)
            EMBED_MODEL = None

def embed_texts(texts: List[str]):
    if EMBED_MODEL is None:
        return None
    return EMBED_MODEL.encode(texts, convert_to_tensor=True)

# Repo fetching (PyGithub)
def fetch_repo_to_temp(github_url: str) -> str:
    if not GITHUB_AVAILABLE:
        raise RuntimeError("PyGithub is not installed. Install pygithub to fetch GitHub repos.")
    # derive owner/repo from URL
    parts = github_url.rstrip('/').split('/')
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL.")
    repo_name = "/".join(parts[-2:])
    g = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
    try:
        repo = g.get_repo(repo_name)
    except GithubException as e:
        raise RuntimeError(f"Could not access repo {repo_name}: {e}")
    tmpdir = tempfile.mkdtemp(prefix="repo_")
    logging.info("Downloading repo %s -> %s", repo_name, tmpdir)
    def download_dir(contents, path):
        for item in contents:
            if item.type == 'dir':
                new_path = os.path.join(path, item.name)
                os.makedirs(new_path, exist_ok=True)
                download_dir(repo.get_contents(item.path), new_path)
            else:
                try:
                    content = item.decoded_content
                    fpath = os.path.join(path, item.name)
                    with open(fpath, 'wb') as f:
                        f.write(content)
                except Exception as e:
                    logging.warning("Failed to write %s: %s", item.path, e)
    try:
        contents = repo.get_contents("")
        download_dir(contents, tmpdir)
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"Failed to download repo: {e}")
    return tmpdir

# Core file processing pipeline ---------------------------------------------
def process_file(file_path: str, lang_hint: Optional[str]=None) -> Dict[str, Any]:
    # read file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
    except Exception as e:
        logging.warning("Could not read file %s: %s", file_path, e)
        return {"path": file_path, "tokens": [], "fingerprints": set(), "token_count": 0, "text": ""}
    lang = lang_hint or "unknown"
    normalized = remove_comments_and_strings(raw, lang)
    normalized = mask_literals(normalized)
    tokens = tokenize_text(normalized)
    shingles = shingle_tokens(tokens)
    hashes = [hash_shingle(s) for s in shingles]
    fingerprints = winnow_hashes(hashes)
    return {
        "path": file_path,
        "tokens": tokens,
        "fingerprints": fingerprints,
        "token_count": len(tokens),
        "text": normalized
    }

def aggregate_repo_similarity(repo_root: str, use_semantic: bool = True, return_pairs: int = 10) -> Dict[str, Any]:
    # detect files
    files_by_lang = detect_files_by_language(repo_root)
    all_files = []
    for lang, files in files_by_lang.items():
        for f in files:
            all_files.append((f, lang))
    if not all_files:
        return {"repo_similarity": 0.0, "reason": "No supported source files found.", "top_pairs": []}

    # Process files
    processed = []
    for f, lang in all_files:
        processed.append(process_file(f, lang_hint=lang))

    # Build MinHash LSH index for fast candidate discovery
    if not DATASKETCH_AVAILABLE:
        raise RuntimeError("datasketch is required for MinHash/LSH. Install datasketch.")
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_PERM)
    minhashes = {}
    for i, p in enumerate(processed):
        m = build_minhash_from_fingerprints(p["fingerprints"])
        minhashes[i] = m
        try:
            lsh.insert(str(i), m)
        except Exception as e:
            logging.warning("LSH insert failed for %s: %s", p["path"], e)

    # Optional semantic embeddings
    semantic_embeddings = {}
    if use_semantic and SBERT_AVAILABLE:
        init_embedding_model()
        try:
            texts = [p["text"] if p["text"] else " " for p in processed]
            embeds = embed_texts(texts)
            for i, emb in enumerate(embeds):
                semantic_embeddings[i] = emb
        except Exception as e:
            logging.warning("Semantic embedding failed: %s", e)
            semantic_embeddings = {}

    # For each file, find best match and score
    file_best_scores = []  # tuple (index, best_score, best_match_index, syntactic, semantic, token_count)
    for i, p in enumerate(processed):
        # candidates from LSH
        try:
            candidates = lsh.query(minhashes[i])
        except Exception:
            candidates = []
        best_score = 0.0
        best_j = None
        best_syn = 0.0
        best_sem = 0.0
        for cid in candidates:
            j = int(cid)
            if i == j:
                continue
            # exact syntactic Jaccard on fingerprints
            a = processed[i]["fingerprints"]
            b = processed[j]["fingerprints"]
            if not a and not b:
                syn_sim = 0.0
            else:
                inter = len(a & b)
                union = len(a | b) if len(a | b) > 0 else 1
                syn_sim = inter / union
            sem_sim = 0.0
            if i in semantic_embeddings and j in semantic_embeddings:
                try:
                    sem_sim = float(sbert_util.pytorch_cos_sim(semantic_embeddings[i], semantic_embeddings[j]).item())
                except Exception:
                    sem_sim = 0.0
            # combine: give more weight to syntactic by default
            alpha = 0.7
            beta = 0.3
            combined = alpha * syn_sim + beta * sem_sim
            if combined > best_score:
                best_score = combined
                best_j = j
                best_syn = syn_sim
                best_sem = sem_sim
        file_best_scores.append((i, best_score, best_j, best_syn, best_sem, processed[i]["token_count"]))

    # Weighted aggregation
    total_tokens = sum(max(1, x[5]) for x in file_best_scores)
    repo_score = 0.0
    for rec in file_best_scores:
        weight = rec[5] / total_tokens
        repo_score += weight * rec[1]

    repo_percent = round(repo_score * 100, 2)

    # Prepare top suspicious pairs for review
    sorted_pairs = sorted([r for r in file_best_scores if r[1] > 0], key=lambda x: x[1], reverse=True)[:return_pairs]
    top_pairs = []
    for (i, score, j, syn, sem, tcount) in sorted_pairs:
        if j is None:
            continue
        top_pairs.append({
            "file_a": processed[i]["path"],
            "file_b": processed[j]["path"],
            "combined_score": round(score, 4),
            "syntactic": round(syn, 4),
            "semantic": round(sem, 4),
            "tokens_in_a": processed[i]["token_count"],
            "tokens_in_b": processed[j]["token_count"]
        })

    return {"repo_similarity": repo_percent, "top_pairs": top_pairs, "num_files": len(processed)}

# Optional: MOSS helper (auxiliary)
def run_moss_on_repo(repo_root: str, language_hint: Optional[str] = "python") -> Optional[str]:
    if not MOSS_AVAILABLE or not MOSS_USER_ID:
        return None
    try:
        m = mosspy.Moss(MOSS_USER_ID, language_hint)
        py_files = glob.glob(os.path.join(repo_root, "**/*.*"), recursive=True)
        # add all text files (MOSS will skip unsupported)
        for f in py_files:
            if is_text_file(f):
                m.addFile(f)
        url = m.send()
        return url
    except Exception as e:
        logging.warning("MOSS run failed: %s", e)
        return None

# FastAPI endpoint ----------------------------------------------------------
class EvalRequest(BaseModel):
    github_url: HttpUrl
    question: str

app = FastAPI(title="Comprehensive Repo Plagiarism Detector", version="1.0")

@app.post("/evaluate/")
async def evaluate(req: EvalRequest):
    # 1. fetch repo
    if not GITHUB_AVAILABLE:
        raise HTTPException(status_code=500, detail="PyGithub not installed on server.")
    try:
        repo_dir = fetch_repo_to_temp(str(req.github_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # 2. run corpus similarity engine
        try:
            result = aggregate_repo_similarity(repo_dir, use_semantic=SBERT_AVAILABLE, return_pairs=10)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Similarity engine failed: {e}")

        # 3. optionally run MOSS as secondary check
        moss_url = None
        try:
            if MOSS_AVAILABLE and MOSS_USER_ID:
                moss_url = run_moss_on_repo(repo_dir)
        except Exception:
            moss_url = None

        # 4. cleanup temp dir (optional: comment out if you want to keep files)
        try:
            shutil.rmtree(repo_dir, ignore_errors=True)
        except Exception:
            pass

        # 5. respond with repository similarity % and evidence
        response = {
            "status": "complete",
            "repository": req.github_url,
            "question": req.question,
            "repo_similarity_percent": result.get("repo_similarity", 0.0),
            "num_files_scanned": result.get("num_files", 0),
            "top_suspicious_pairs": result.get("top_pairs", []),
            "moss_report_url": moss_url
        }
        return response

    finally:
        # ensure cleanup if any unexpected issue
        if os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception:
                pass

@app.get("/")
def root():
    return {"message": "Plagiarism Detector up. POST /evaluate/ with {github_url, question}."}

