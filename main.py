import os
import re
import io
import math
import glob
import shutil
import hashlib
import tempfile
import logging
import json
from typing import List, Tuple, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv


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


# Optional Vertex AI integration
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    VERTEX_AI_AVAILABLE = True
except Exception:
    VERTEX_AI_AVAILABLE = False


# Alternative: try Google AI SDK if Vertex AI not available
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except Exception:
    GOOGLE_AI_AVAILABLE = False


# CONFIG
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
MOSS_USER_ID = os.getenv("MOSS_USER_ID", None)  # optional
VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Alternative to Vertex AI


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
    binary_exts = {'.png', '.jpg', '.jpeg', '.gif', '.class', '.exe', '.dll', '.so', '.o', '.pdf'}
    ext = os.path.splitext(path)[1].lower()
    if ext in binary_exts:
        return False
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


def remove_comments_and_strings(text: str, lang: str) -> str:
    if lang == 'python':
        text = re.sub(r'("""|\'\'\')(?:.|[\r\n])*?\1', ' <STR> ', text, flags=re.MULTILINE)
    text = re.sub(r'("([^"\\]|\\.)*")|(\'([^\'\\]|\\.)*\')', ' <STR> ', text)
    text = re.sub(r'//.*', ' ', text)
    text = re.sub(r'#.*', ' ', text)
    text = re.sub(r'/\*(?:.|[\r\n])*?\*/', ' ', text)
    text = re.sub(r'""".*?"""', ' ', text, flags=re.S)
    return text


def mask_literals(text: str) -> str:
    text = re.sub(r'\b\d+(\.\d+)?\b', ' <NUM> ', text)
    return text


def tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r'[A-Za-z_]\w*|==|!=|<=|>=|->|::|<=|>=|[{}()\[\];.,:+\-*/%&|^~<>]', text)
    return tokens


def shingle_tokens(tokens: List[str], k: int = SHINGLE_K) -> List[str]:
    if len(tokens) < k:
        return [' '.join(tokens)]
    return [' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1)]


def hash_shingle(sh: str) -> int:
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
    if len(hashes) < w:
        fingerprints.update(hashes)
    return fingerprints


def build_minhash_from_fingerprints(fp_set: set, num_perm: int = MINHASH_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for fp in fp_set:
        m.update(str(fp).encode('utf8'))
    return m


EMBED_MODEL = None
def init_embedding_model():
    global EMBED_MODEL
    if SBERT_AVAILABLE and EMBED_MODEL is None:
        try:
            EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.warning("SentenceTransformer init failed: %s", e)
            EMBED_MODEL = None


def embed_texts(texts: List[str]):
    if EMBED_MODEL is None:
        return None
    return EMBED_MODEL.encode(texts, convert_to_tensor=True)


def fetch_repo_to_temp(github_url: str) -> str:
    if not GITHUB_AVAILABLE:
        raise RuntimeError("PyGithub is not installed. Install pygithub to fetch GitHub repos.")
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


def analyze_repo_structure(github_url: str) -> Dict[str, Any]:
    if not GITHUB_AVAILABLE:
        return {"error": "PyGithub not available"}
    try:
        parts = github_url.rstrip('/').split('/')
        if len(parts) < 2:
            raise ValueError("Invalid GitHub URL.")
        repo_name = "/".join(parts[-2:])
        g = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
        repo = g.get_repo(repo_name)
        repo_data = {
            "name": repo.name,
            "description": repo.description or "No description",
            "language": repo.language or "Unknown",
            "size": repo.size,
            "created_at": str(repo.created_at),
            "updated_at": str(repo.updated_at),
            "stargazers_count": repo.stargazers_count,
            "forks_count": repo.forks_count,
            "open_issues_count": repo.open_issues_count,
            "topics": repo.get_topics(),
            "has_wiki": repo.has_wiki,
            "has_pages": repo.has_pages,
            "has_downloads": repo.has_downloads,
        }
        file_structure = []
        readme_content = ""
        try:
            contents = repo.get_contents("")
            for item in contents:
                if item.type == "file":
                    file_info = {
                        "name": item.name,
                        "path": item.path,
                        "size": item.size,
                        "type": "file"
                    }
                    file_structure.append(file_info)
                    if item.name.lower().startswith('readme'):
                        try:
                            readme_content = item.decoded_content.decode('utf-8')[:2000]
                        except Exception:
                            readme_content = "README file exists but content not accessible"
                elif item.type == "dir":
                    file_structure.append({
                        "name": item.name,
                        "path": item.path,
                        "type": "directory"
                    })
        except Exception as e:
            logging.warning("Failed to analyze repo structure: %s", e)
            file_structure = [{"error": str(e)}]
        commits_count = 0
        try:
            commits = repo.get_commits()
            commits_count = commits.totalCount if hasattr(commits, 'totalCount') else len(list(commits)[:100])
        except Exception:
            commits_count = 0
        repo_data.update({
            "file_structure": file_structure[:20],
            "readme_content": readme_content,
            "commits_count": commits_count
        })
        return repo_data
    except Exception as e:
        logging.error("Repository analysis failed: %s", e)
        return {"error": str(e)}


def calculate_assessment_completeness(repo_data: Dict[str, Any], question: str) -> float:
    if "error" in repo_data:
        return 0.0
    def clamp_text(text: str, max_chars: int = 4000) -> str:
        if not text:
            return ""
        return text[:max_chars]
    file_rows = repo_data.get("file_structure", [])
    file_rows_compact = []
    for r in file_rows[:50]:
        file_rows_compact.append({
            "name": r.get("name"),
            "path": r.get("path"),
            "type": r.get("type"),
            "size": r.get("size", None),
        })
    readme_text = clamp_text(repo_data.get("readme_content", "") or "")
    analysis_prompt = f"""
You are an expert code reviewer and assessment evaluator. Evaluate how complete this GitHub repository is as a submission for the specified assessment.

Assessment (task name): {question}

Repository:
- Name: {repo_data.get('name', 'N/A')}
- Description: {repo_data.get('description', 'N/A')}
- Primary Language: {repo_data.get('language', 'N/A')}
- Size(KB): {repo_data.get('size', 0)}
- Files Listed: {len(file_rows_compact)}
- Commits: {repo_data.get('commits_count', 0)}
- Has Wiki: {repo_data.get('has_wiki', False)}
- Topics: {', '.join(repo_data.get('topics', []))}

File Structure (sampled):
{json.dumps(file_rows_compact, indent=2)}

README (truncated):
{readme_text}

Scoring criteria:
1) Code structure/organization & essential files.
2) Documentation sufficiency.
3) Evident complexity/functionality.
4) Development practices.
5) Alignment with task.

Return ONLY a single number (0-100) representing completeness %.
""".strip()
    def extract_percentage(text: str) -> Optional[float]:
        try:
            m = re.search(r'\d+(?:\.\d+)?', text or "")
            if not m:
                return None
            val = float(m.group())
            return max(0.0, min(100.0, val))
        except Exception:
            return None
    if VERTEX_AI_AVAILABLE and VERTEX_AI_PROJECT_ID:
        try:
            vertexai.init(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_LOCATION or "us-central1")
            model = GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(analysis_prompt)
            text = (getattr(resp, "text", None) or "").strip()
            score = extract_percentage(text)
            if score is not None: return score
        except Exception as e:
            logging.warning("Vertex AI analysis failed: %s", e)
    if GOOGLE_AI_AVAILABLE and GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(analysis_prompt)
            text = (getattr(resp, "text", None) or "").strip()
            score = extract_percentage(text)
            if score is not None: return score
        except Exception as e:
            logging.warning("Google AI analysis failed: %s", e)
    logging.info("AI analysis not available or failed. Using heuristic scoring.")
    return calculate_assessment_completeness_heuristic(repo_data)


def calculate_assessment_completeness_heuristic(repo_data: Dict[str, Any]) -> float:
    score = 0.0
    files = repo_data.get('file_structure', [])
    if len(files) > 0: score += 10
    if len(files) > 5: score += 10
    if any(f['name'].lower().startswith('readme') for f in files): score += 10
    if any(f['name'].endswith(('.py', '.js', '.java', '.cpp', '.c')) for f in files): score += 10
    if repo_data.get('description'): score += 10
    if repo_data.get('commits_count', 0) > 1: score += 10
    if repo_data.get('commits_count', 0) > 5: score += 10
    size = repo_data.get('size', 0)
    if size > 10: score += 10
    if size > 100: score += 10
    if repo_data.get('readme_content') and len(repo_data.get('readme_content', '')) > 100: score += 10
    return min(100.0, score)


# ---------- NEW: Final Score Calculation ----------
def calculate_final_score(similarity_percent: float, completeness_percent: float) -> float:
    """Final score: 70% completeness, minus 30% plagiarism penalty."""
    plagiarism_penalty = similarity_percent * 0.3
    base_score = (completeness_percent * 0.7) - plagiarism_penalty
    return round(max(0.0, min(100.0, base_score)), 2)
# ---------------------------------------------------


def process_file(file_path: str, lang_hint: Optional[str]=None) -> Dict[str, Any]:
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
    files_by_lang = detect_files_by_language(repo_root)
    all_files = [(f, lang) for lang, files in files_by_lang.items() for f in files]
    if not all_files:
        return {"repo_similarity": 0.0, "reason": "No supported source files found.", "top_pairs": []}
    processed = [process_file(f, lang_hint=lang) for f, lang in all_files]
    if not DATASKETCH_AVAILABLE:
        raise RuntimeError("datasketch is required for MinHash/LSH.")
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_PERM)
    minhashes = {}
    for i, p in enumerate(processed):
        m = build_minhash_from_fingerprints(p["fingerprints"])
        minhashes[i] = m
        try:
            lsh.insert(str(i), m)
        except Exception as e:
            logging.warning("LSH insert failed for %s: %s", p["path"], e)
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
    file_best_scores = []
    for i, p in enumerate(processed):
        try:
            candidates = lsh.query(minhashes[i])
        except Exception:
            candidates = []
        best_score = 0.0; best_j=None; best_syn=0.0; best_sem=0.0
        for cid in candidates:
            j = int(cid)
            if i == j: continue
            a = processed[i]["fingerprints"]
            b = processed[j]["fingerprints"]
            syn_sim = (len(a & b) / len(a | b)) if a or b else 0.0
            sem_sim = 0.0
            if i in semantic_embeddings and j in semantic_embeddings:
                try:
                    sem_sim = float(sbert_util.pytorch_cos_sim(semantic_embeddings[i], semantic_embeddings[j]).item())
                except Exception:
                    sem_sim = 0.0
            combined = 0.7 * syn_sim + 0.3 * sem_sim
            if combined > best_score:
                best_score, best_j, best_syn, best_sem = combined, j, syn_sim, sem_sim
        file_best_scores.append((i, best_score, best_j, best_syn, best_sem, processed[i]["token_count"]))
    total_tokens = sum(max(1, x[5]) for x in file_best_scores)
    repo_score = sum((rec[1] * (rec[5]/total_tokens)) for rec in file_best_scores)
    repo_percent = round(repo_score * 100, 2)
    sorted_pairs = sorted([r for r in file_best_scores if r[1] > 0], key=lambda x: x[1], reverse=True)[:return_pairs]
    top_pairs = []
    for (i, score, j, syn, sem, tcount) in sorted_pairs:
        if j is None: continue
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


def run_moss_on_repo(repo_root: str, language_hint: Optional[str] = "python") -> Optional[str]:
    if not MOSS_AVAILABLE or not MOSS_USER_ID:
        return None
    try:
        m = mosspy.Moss(MOSS_USER_ID, language_hint)
        py_files = glob.glob(os.path.join(repo_root, "**/*.*"), recursive=True)
        for f in py_files:
            if is_text_file(f):
                m.addFile(f)
        url = m.send()
        return url
    except Exception as e:
        logging.warning("MOSS run failed: %s", e)
        return None


class EvalRequest(BaseModel):
    github_url: HttpUrl
    question: str


app = FastAPI(title="Comprehensive Repo Plagiarism Detector", version="1.0")


@app.post("/evaluate/")
async def evaluate(req: EvalRequest):
    if not GITHUB_AVAILABLE:
        raise HTTPException(status_code=500, detail="PyGithub not installed.")
    try:
        repo_dir = fetch_repo_to_temp(str(req.github_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        try:
            result = aggregate_repo_similarity(repo_dir, use_semantic=SBERT_AVAILABLE, return_pairs=10)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Similarity engine failed: {e}")
        moss_url = None
        try:
            if MOSS_AVAILABLE and MOSS_USER_ID:
                moss_url = run_moss_on_repo(repo_dir)
        except Exception:
            moss_url = None
        assessment_completeness = 0.0
        repo_data = None
        try:
            logging.info("Starting assessment completeness analysis...")
            repo_data = analyze_repo_structure(str(req.github_url))
            assessment_completeness = calculate_assessment_completeness(repo_data, req.question)
            logging.info("Assessment completeness: %.2f%%", assessment_completeness)
        except Exception as e:
            logging.error("Assessment completeness analysis failed: %s", e)
            assessment_completeness = 0.0
        # NEW: Final score computation
        final_score = calculate_final_score(
            similarity_percent=result.get("repo_similarity", 0.0),
            completeness_percent=assessment_completeness
        )
        try:
            shutil.rmtree(repo_dir, ignore_errors=True)
        except Exception:
            pass
        response = {
            "repository": req.github_url,
            "question": req.question,
            "repo_similarity_percent": result.get("repo_similarity", 0.0),
            "assessment_completeness_percentage": assessment_completeness,
            "final_score": final_score,
            "moss_report_url": moss_url
        }
        return response
    finally:
        if os.path.exists(repo_dir):
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception:
                pass


@app.get("/")
def root():
    return {"message": "Plagiarism Detector up. POST /evaluate/ with {github_url, question}."}
