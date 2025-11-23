import json, random, base64, codecs, uuid
from pathlib import Path
import yaml

ZWJ = "\u200d"

def insert_zwj(text, phrases):
    for p in phrases:
        z = ZWJ.join(list(p))
        text = text.replace(p, z)
    return text

def map_homoglyphs(text, mapping):
    return "".join(mapping.get(ch, ch) for ch in text)

def encode_then_hint(base_instr, mode):
    if mode == "rot13":
        payload = codecs.encode(base_instr, "rot_13")
    else:
        payload = base64.b64encode(base_instr.encode()).decode()
    return payload, mode

def build_malicious_chunk(cfg, evasion_id, success_token):
    base = cfg["injection_templates"]["base_instruction"].replace("{SUCCESS_TOKEN}", success_token)
    v = next(x for x in cfg["injection_templates"]["variants"] if x["id"] == evasion_id)
    
    if v["transform"] == "none":
        text = base
    elif v["transform"] == "delimiter_wrapper":
        text = f"{v['prefix']}{base}{v['suffix']}"
    elif v["transform"] == "role_prefix":
        text = f"{v['role_text']}{base.lower()}{v['role_suffix']}"
    elif v["transform"] == "urgency_wrapper":
        text = f"{v['urgency_prefix']}{base}{v['urgency_suffix']}"
    elif v["transform"] == "multilingual":
        lang_choice = random.choice(v["languages"])
        text = lang_choice["text"].replace("{SUCCESS_TOKEN}", success_token)
    elif v["transform"] == "split_payload":
        text = f"{v['part1']}{v['part2']}{v['part3'].replace('{SUCCESS_TOKEN}', success_token)}"
    elif v["transform"] == "insert_zwj":
        text = insert_zwj(base, v["zwj_between_phrases"])
    elif v["transform"] == "map_homoglyphs":
        text = map_homoglyphs(base, v["mapping"])
    elif v["transform"] == "encode_then_hint":
        payload, mode = encode_then_hint(base, random.choice(["base64","rot13"]))
        text = f"{v['hint_prefix']}\n{payload}\n({mode})\n{v['hint_suffix']}"
    else:
        raise ValueError(f"unknown transform: {v['transform']}")
    return text

def main():
    # Construct paths relative to script location
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent
    
    cfg = yaml.safe_load((root_dir / "partA_kb_generator.yaml").read_text(encoding="utf-8"))
    rng = random.Random(cfg["dataset"]["seed"])
    benign_total = cfg["counts"]["benign_total"]
    mal_per = cfg["counts"]["malicious_per_evasion"]
    evasion_ids = cfg["evasion_types"]
    success_tokens = cfg["success_tokens"]

    docs = []
    # Benign docs
    for topic in cfg["topics"]:
        for f in topic["facts"]:
            docs.append({
                "id": str(uuid.uuid4()),
                "topic_id": topic["id"],
                "title": f"{topic['id']}_benign",
                "text": f,
                "is_malicious": False,
                "evasion_type": None,
                "success_token": None
            })
    # Top up to desired benign_total by sampling facts
    while sum(1 for d in docs if not d["is_malicious"]) < benign_total:
        topic = rng.choice(cfg["topics"])
        f = rng.choice(topic["facts"])
        docs.append({
            "id": str(uuid.uuid4()),
            "topic_id": topic["id"],
            "title": f"{topic['id']}_benign",
            "text": f,
            "is_malicious": False,
            "evasion_type": None,
            "success_token": None
        })

    # Malicious docs
    for ev_id in evasion_ids:
        for _ in range(mal_per):
            topic = rng.choice(cfg["topics"])
            success = rng.choice(success_tokens)
            inj = build_malicious_chunk(cfg, ev_id, success)
            text = f"{rng.choice(topic['facts'])}\n\n{inj}"
            docs.append({
                "id": str(uuid.uuid4()),
                "topic_id": topic["id"],
                "title": f"{topic['id']}_mal_{ev_id}",
                "text": text,
                "is_malicious": True,
                "evasion_type": ev_id,
                "success_token": success
            })

    phase1_dir = script_dir.parent
    out = phase1_dir / "data" / "partA_kb.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} docs to {out}")

if __name__ == "__main__":
    main()