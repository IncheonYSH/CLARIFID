import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer
from collections import OrderedDict
import re
from collections import defaultdict

# --------------------------------------------------------------------------
# 1) CheXbert Model Definition & Loading
# --------------------------------------------------------------------------

class bert_labeler(nn.Module):
    """
    Minimal CheXbert model definition.
    For more complex usage or training, adapt as needed.
    """
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        super(bert_labeler, self).__init__()
        if pretrain_path is not None:
            # If you have a local pretrained path
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            # If you wanted Bio_ClinicalBERT, you'd do something else here
            # But let's ignore that or set it to False for now
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        hidden_size = self.bert.pooler.dense.in_features

        # 13 heads with 4 classes each => present, absent, uncertain, not mentioned
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # 1 head for "No Finding" => 2 classes
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, input_ids, attention_mask):
        final_hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)

        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))  # shape e.g. [B,4] or [B,2]
        return out


def load_chexbert_model_and_tokenizer(checkpoint_path, device="cpu"):
    """
    Load the CheXbert model weights from a training checkpoint that likely has:
      {
        'epoch': ...,
        'model_state_dict': { ... actual weights ... },
        'optimizer_state_dict': ...
      }
    Remap any 'module.*' keys then load into bert_labeler. Returns (model, tokenizer).
    """
    # 1) Initialize the model
    model = bert_labeler()

    # 2) Load the full checkpoint (which might contain extra keys)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 3) Remap 'module.*' keys to match the single-GPU model definition
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if "module.bert." in key:
            new_key = key.replace("module.bert.", "bert.")
        elif "module.linear_heads." in key:
            new_key = key.replace("module.linear_heads.", "linear_heads.")
        elif "module." in key:
            new_key = key.replace("module.", "")
        else:
            new_key = key
        new_state_dict[new_key] = val

    # 4) Load into our model
    model.load_state_dict(new_state_dict, strict=False)

    # 5) Eval mode, move to device
    model.eval()
    model.to(device)

    # 6) Use the standard bert-base-uncased tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer


@torch.no_grad()
def get_chexbert_predictions(text, model, tokenizer, device="cpu"):
    """
    Given a single string 'text', returns a list of 14 class predictions for CheXbert:
      - Indices 0..12 => each in {0,1,2,3} (Absent, Present, Uncertain, Not Mentioned)
      - Index 13 => in {0,1} for "No Finding" (0=No, 1=Yes)
    """
    text = text.strip()
    if not text:
        return None  # or any default

    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)

    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    preds = []
    for logits in outputs:
        pred_class = torch.argmax(logits, dim=1).item()  # single int
        preds.append(pred_class)
    return preds

# --------------------------------------------------------------------------
# 2) Text Parsing with section_parser logic
#    (Extract FINDINGS, IMPRESSION, LAST_PARAGRAPH, etc.)
# --------------------------------------------------------------------------

def list_rindex(lst, val):
    """Helper function to find the last index of 'val' in a list."""
    return len(lst) - 1 - lst[::-1].index(val)

# You can put the content of `section_parser.py` here or import it.
# For brevity, we'll include the relevant functions inline.

def normalize_section_names(section_names):
    # Same logic as your section_parser
    frequent_sections = {
        "preamble": "preamble",
        "impression": "impression",
        "comparison": "comparison",
        "indication": "indication",
        "findings": "findings",
        "examination": "examination",
        "technique": "technique",
        "history": "history",
        "comparisons": "comparison",
        "clinical history": "history",
        "reason for examination": "indication",
        "reason for exam": "indication",
        "clinical information": "history",
        "exam": "examination",
        "clinical indication": "indication",
        "conclusion": "impression",
        "chest, two views": "findings",
        "recommendation(s)": "recommendations",
        "type of examination": "examination",
        "reference exam": "comparison",
        "patient history": "history",
        "addendum": "addendum",
        "comparison exam": "comparison",
        "date": "date",
        "comment": "comment",
        "findings and impression": "impression",
        "wet read": "wet read",
        "comparison film": "comparison",
        "recommendations": "recommendations",
        "findings/impression": "impression",
        "pfi": "history",
        'recommendation': 'recommendations',
        'wetread': 'wet read',
        'ndication': 'impression',
        'impresson': 'impression',
        'imprression': 'impression',
        'imoression': 'impression',
        'impressoin': 'impression',
        'imprssion': 'impression',
        'impresion': 'impression',
        'imperssion': 'impression',
        'mpression': 'impression',
        'impession': 'impression',
        'findings/ impression': 'impression',
        'finding': 'findings',
        'findins': 'findings',
        'findindgs': 'findings',
        'findgings': 'findings',
        'findngs': 'findings',
        'findnings': 'findings',
        'finidngs': 'findings',
        'idication': 'indication',
        'reference findings': 'findings',
        'comparision': 'comparison',
        'comparsion': 'comparison',
        'comparrison': 'comparison',
        'comparisions': 'comparison'
    }

    p_findings_keywords = [
        'chest',
        'portable',
        'pa and lateral',
        'lateral and pa',
        'ap and lateral',
        'lateral and ap',
        'frontal and',
        'two views',
        'frontal view',
        'pa view',
        'ap view',
        'one view',
        'lateral view',
        'bone window',
        'frontal upright',
        'frontal semi-upright',
        'ribs',
        'pa and lat'
    ]
    p_findings = re.compile('({})'.format('|'.join(p_findings_keywords)))

    main_sections = ['impression', 'findings', 'history', 'comparison', 'addendum']

    out_names = []
    for s in section_names:
        s_lower = s.lower().strip()
        if s_lower in frequent_sections:
            out_names.append(frequent_sections[s_lower])
            continue
        # Check partial matches
        matched_main = False
        for m in main_sections:
            if m in s_lower:
                out_names.append(m)
                matched_main = True
                break
        if matched_main:
            continue
        # Check if the section name looks like a 'findings' chunk
        if p_findings.search(s_lower) is not None:
            out_names.append('findings')
        else:
            # default if we can't map
            out_names.append(s_lower)
    return out_names


def section_text(text):
    """
    Splits text into sections based on capitalized headings.
    Returns (sections, section_names, section_idx).
    """
    p_section = re.compile(r'\n ([A-Z ()/,-]+):\s', re.DOTALL)

    sections = []
    section_names = []
    section_idx = []

    idx = 0
    s = p_section.search(text, idx)
    if s:
        # The text before the first match is 'preamble'
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1)
            idx_start = s.end()
            # skip past the first newline if it exists
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0
            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)
    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)

    # Normalize
    section_names = normalize_section_names(section_names)

    # Remove any empty "impression" or "findings" sections
    # that sometimes occur if they have no content
    for i in reversed(range(len(section_names))):
        if section_names[i] in ('impression', 'findings'):
            if sections[i].strip() == '':
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    # If no impression/findings, we can label the last paragraph as 'last_paragraph'
    # if there's a double newline near the end.
    # (This is the same logic as the original snippet.)
    if ('impression' not in section_names) and ('findings' not in section_names):
        # Create a new "last_paragraph" if there's a chunk near the end
        # but only if it has a double-newline or something that indicates
        # a separate chunk of text.
        # You can refine this condition if needed:
        if '\n \n' in sections[-1]:
            chunked = sections[-1].split('\n \n', 1)
            sections[-1] = chunked[0]
            last_p = chunked[1] if len(chunked) > 1 else ''
            sections.append(last_p)
            section_names.append('last_paragraph')
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx


def custom_mimic_cxr_rules():
    """
    Returns custom_section_names, custom_indices for special-case parsing.
    These come from your script snippet.
    """
    custom_section_names = {
        's50913680': 'recommendations',
        's59363654': 'examination',
        's59279892': 'technique',
        's59768032': 'recommendations',
        's57936451': 'indication',
        's50058765': 'indication',
        's53356173': 'examination',
        's53202765': 'technique',
        's50808053': 'technique',
        's51966317': 'indication',
        's50743547': 'examination',
        's56451190': 'note',
        's59067458': 'recommendations',
        's59215320': 'examination',
        's55124749': 'indication',
        's54365831': 'indication',
        's59087630': 'recommendations',
        's58157373': 'recommendations',
        's56482935': 'recommendations',
        's58375018': 'recommendations',
        's54654948': 'indication',
        's55157853': 'examination',
        's51491012': 'history',
    }

    custom_indices = {
        's50525523': [201, 349],
        's57564132': [233, 554],
        's59982525': [313, 717],
        's53488209': [149, 475],
        's54875119': [234, 988],
        's50196495': [59, 399],
        's56579911': [59, 218],
        's52648681': [292, 631],
        's59889364': [172, 453],
        's53514462': [73, 377],
        's59505494': [59, 450],
        's53182247': [59, 412],
        's51410602': [47, 320],
        's56412866': [522, 822],
        's54986978': [59, 306],
        's59003148': [262, 505],
        's57150433': [61, 394],
        's56760320': [219, 457],
        's59562049': [158, 348],
        's52674888': [145, 296],
        's55258338': [192, 568],
        's59330497': [140, 655],
        's52119491': [179, 454],
        # below have no findings at all
        's58235663': [0, 0],
        's50798377': [0, 0],
        's54168089': [0, 0],
        's53071062': [0, 0],
        's56724958': [0, 0],
        's54231141': [0, 0],
        's53607029': [0, 0],
        's52035334': [0, 0],
    }

    return custom_section_names, custom_indices


def parse_mimic_cxr_report(text_file):
    """
    Parse a single MIMIC-CXR text file into:
      {
        "FINDINGS": str or "",
        "IMPRESSION": str or "",
        "LAST_PARAGRAPH": str or "",
        "COMPARISON": str or "",
        "all_text": str
      }
    using the section_parser logic and custom rules.
    """
    if (not text_file) or (not os.path.exists(text_file)):
        # Return empty placeholders
        return {
            "FINDINGS": "",
            "IMPRESSION": "",
            "LAST_PARAGRAPH": "",
            "COMPARISON": "",
            "all_text": ""
        }

    # Load custom rules
    c_names, c_indices = custom_mimic_cxr_rules()

    # Read the full report text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text_all = text  # keep original
    s_stem = os.path.splitext(os.path.basename(text_file))[0]  # e.g. 's12345678'

    # 1) Check if there's a custom slice for this study
    if s_stem in c_indices:
        start_i, end_i = c_indices[s_stem]
        # safe-guard
        start_i = max(0, min(len(text), start_i))
        end_i = max(0, min(len(text), end_i))
        text = text[start_i:end_i]

    # 2) Parse sections
    sections, section_names, section_idx = section_text(text)

    # If there's a custom "section name" we want to use
    # (this indicates the actual text is stored in that named section)
    # e.g. if s_stem -> 'recommendations', we take the last occurrence
    if s_stem in c_names:
        forced_section = c_names[s_stem]
        if forced_section in section_names:
            idx = list_rindex(section_names, forced_section)
            # entire text = that forced section
            # but let's still parse out the others if they exist
            override_text = sections[idx].strip()
            # We can skip or keep the rest, depending on your preference
            # For simplicity, let's just *overwrite* 'text' with that section
            text = override_text
            # Re-run the parser on just that subset (optional)
            sections, section_names, section_idx = section_text(text)

    # Build dictionary of the sections we care about:
    out_data = {
        "FINDINGS": "",
        "IMPRESSION": "",
        "LAST_PARAGRAPH": "",
        "COMPARISON": "",
        "all_text": text_all  # original entire text
    }

    # We capture the last occurrence of each (if present)
    for sec_name in ["impression", "findings", "last_paragraph", "comparison"]:
        if sec_name in section_names:
            idx = list_rindex(section_names, sec_name)
            out_data[sec_name.upper()] = sections[idx].strip()

    def sanitize_text(t: str) -> str:
        # Lowercase
        # t = t.lower()
        # Replace newlines with spaces
        t = t.replace("\n", " ")
        # Collapse multiple spaces/tabs into one space
        t = re.sub(r"\s+", " ", t)
        # Strip leading/trailing
        t = t.strip()
        return t

    out_data["FINDINGS"] = sanitize_text(out_data["FINDINGS"])
    out_data["IMPRESSION"] = sanitize_text(out_data["IMPRESSION"])
    out_data["LAST_PARAGRAPH"] = sanitize_text(out_data["LAST_PARAGRAPH"])
    out_data["COMPARISON"] = sanitize_text(out_data["COMPARISON"])
    out_data["all_text"] = sanitize_text(out_data["all_text"])

    # Return them with uppercase keys
    return {
        "FINDINGS": out_data["FINDINGS"],
        "IMPRESSION": out_data["IMPRESSION"],
        "LAST_PARAGRAPH": out_data["LAST_PARAGRAPH"],
        "COMPARISON": out_data["COMPARISON"],
        "all_text": out_data["all_text"]
    }

# --------------------------------------------------------------------------
# 3) Token Counting
# --------------------------------------------------------------------------

def count_gpt2_tokens(text: str, tokenizer: GPT2Tokenizer) -> int:
    """Count the number of tokens in a text using GPT-2 tokenizer."""
    if not text.strip():
        return 0
    return len(tokenizer.encode(text))

# --------------------------------------------------------------------------
# 4) Main CSV creation
# --------------------------------------------------------------------------

CHEXPERT_LABEL_ORDER = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding"
]


def create_mimic_study_level_metadata_csv(
    image_root,
    text_root,
    split_csv,
    chexpert_csv,
    meta_csv,
    out_csv,
    chexbert_ckpt="/path/to/chexbert.pth",
    device="cpu"
):
    """
    Builds a single record PER STUDY, then writes them out as a CSV file.

    CSV columns include:
      subject_id, study_id, split, image_paths,
      findings, impression, last_paragraph, report_all,
      has_findings, has_impression, has_last_paragraph,
      findings_tokens_gpt2, impression_tokens_gpt2, last_paragraph_tokens_gpt2, all_text_tokens_gpt2,
      chexpert_labels, chexbert_labels,
      chex_label_diff, num_images
    """
    # 0) Load CheXbert model & tokenizer and GPT-2 tokenizer
    chexbert_model, chexbert_tokenizer = load_chexbert_model_and_tokenizer(
        checkpoint_path=chexbert_ckpt,
        device=device
    )
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 1) Read splits CSV
    df_split = pd.read_csv(split_csv)
    split_dict = {}
    for _, row in df_split.iterrows():
        key_3 = (int(row["subject_id"]), int(row["study_id"]), str(row["dicom_id"]))
        split_dict[key_3] = str(row["split"]).lower()

    # 2) Read CheXpert CSV
    df_chex = pd.read_csv(chexpert_csv)
    chexpert_dict_raw = {}
    for _, row in df_chex.iterrows():
        subj, stdy = int(row["subject_id"]), int(row["study_id"])
        label_data = {}
        for label in CHEXPERT_LABEL_ORDER:
            label_data[label] = row.get(label, None)
        chexpert_dict_raw[(subj, stdy)] = label_data

    def convert_labels_to_list(label_data):
        out = []
        for lbl in CHEXPERT_LABEL_ORDER:
            val = label_data.get(lbl, 0)
            if pd.isnull(val):
                val = 0
            out.append(int(val))
        return out

    # 3) Read metadata CSV
    df_meta = pd.read_csv(meta_csv)
    meta_dict = {}
    for _, row in df_meta.iterrows():
        key_3 = (int(row["subject_id"]), int(row["study_id"]), str(row["dicom_id"]))
        meta_dict[key_3] = {
            "ViewPosition": row.get("ViewPosition", None),
            "PerformedProcedureStepDescription": row.get("PerformedProcedureStepDescription", None)
        }

    # 4) Gather records
    study_dict = {}
    no_imp_find = 0
    no_imp = 0
    no_find = 0
    no_last = 0  # track missing last_paragraph, if desired

    for subdir, _, files in os.walk(image_root):
        jpgs = [f for f in files if f.lower().endswith(".jpg")]
        if not jpgs:
            continue

        rel_subdir = os.path.relpath(subdir, image_root)
        parts = rel_subdir.split(os.sep)
        if len(parts) < 3:
            continue

        patient_folder = parts[-2]  # e.g. "p10000032"
        study_folder = parts[-1]    # e.g. "s50414267"
        if patient_folder.startswith("p"):
            patient_id = int(patient_folder[1:])
        else:
            patient_id = int(patient_folder)
        if study_folder.startswith("s"):
            study_id = int(study_folder[1:])
        else:
            study_id = int(study_folder)

        # Parse text via our new function
        text_file = os.path.join(text_root, rel_subdir + ".txt")
        text_info = parse_mimic_cxr_report(text_file)
        findings_text = text_info.get("FINDINGS", "")
        impression_text = text_info.get("IMPRESSION", "")
        last_para_text = text_info.get("LAST_PARAGRAPH", "")
        all_text = text_info.get("all_text", "")

        # Track empties
        if not impression_text and not findings_text:
            no_imp_find += 1
        if not impression_text:
            no_imp += 1
        if not findings_text:
            no_find += 1
        if not last_para_text:
            no_last += 1

        # Official CheXpert label => array of 14
        raw_labels = chexpert_dict_raw.get((patient_id, study_id), {})
        chex_labels_list = convert_labels_to_list(raw_labels)

        # Collect DICOMs for this study
        dicoms_for_this_study = []
        for f in jpgs:
            dicom_id = os.path.splitext(f)[0]
            key_3 = (patient_id, study_id, dicom_id)
            if key_3 not in split_dict:
                continue
            full_img_path = os.path.join(subdir, f)
            meta_info = meta_dict.get(key_3, {})
            dicoms_for_this_study.append({
                "dicom_id": dicom_id,
                "view_position": meta_info.get("ViewPosition", None),
                "image_path": full_img_path
            })

        # If no matching DICOMs, skip
        if not dicoms_for_this_study:
            continue

        # Get split from the first DICOM
        first_key_3 = (patient_id, study_id, dicoms_for_this_study[0]["dicom_id"])
        split_name = split_dict[first_key_3]

        # -------------------------
        #  Use impression if available, else findings, else last_paragraph, else entire report
        # -------------------------
        if impression_text.strip():
            chexbert_text = impression_text
        elif findings_text.strip():
            chexbert_text = findings_text
        elif last_para_text.strip():
            chexbert_text = last_para_text
        else:
            chexbert_text = all_text

        # CheXbert inference (14-element list)
        chexbert_pred = get_chexbert_predictions(
            chexbert_text, chexbert_model, chexbert_tokenizer, device=device
        )
        if chexbert_pred is None:
            # default "not mentioned" (3) for 0..12, and "No" (0) for "No Finding"
            chexbert_pred = [3]*13 + [0]

        # Compare "presence" (==1) in CheXpert vs CheXbert
        discrepancy = 0
        for i in range(14):
            chexpert_is_present = (chex_labels_list[i] == 1)
            chexbert_is_present = (chexbert_pred[i] == 1)
            if chexpert_is_present != chexbert_is_present:
                discrepancy = 1
                break

        # Count tokens using GPT-2 tokenizer
        findings_tokens = count_gpt2_tokens(findings_text, gpt2_tokenizer)
        impression_tokens = count_gpt2_tokens(impression_text, gpt2_tokenizer)
        lastpara_tokens = count_gpt2_tokens(last_para_text, gpt2_tokenizer)
        all_text_tokens = count_gpt2_tokens(all_text, gpt2_tokenizer)

        # Flatten image paths
        image_paths_joined = ";".join([dic["image_path"] for dic in dicoms_for_this_study])

        # Convert both label lists to semicolon-separated strings
        chexpert_labels_str = ";".join(str(x) for x in chex_labels_list)
        chexbert_labels_str = ";".join(str(x) for x in chexbert_pred)

        # Build final record for CSV
        study_dict[(patient_id, study_id)] = {
            "subject_id": patient_id,
            "study_id": study_id,
            "split": split_name,
            "image_paths": image_paths_joined,

            "findings": findings_text,
            "impression": impression_text,
            "last_paragraph": last_para_text,
            "report_all": all_text,

            "has_findings": 1 if findings_text.strip() else 0,
            "has_impression": 1 if impression_text.strip() else 0,
            "has_last_paragraph": 1 if last_para_text.strip() else 0,

            # Token counts (GPT-2)
            "findings_tokens_gpt2": findings_tokens,
            "impression_tokens_gpt2": impression_tokens,
            "last_paragraph_tokens_gpt2": lastpara_tokens,
            "all_text_tokens_gpt2": all_text_tokens,

            # CheXpert & CheXbert predictions
            "chexpert_labels": chexpert_labels_str,
            "chexbert_labels": chexbert_labels_str,

            "chex_label_diff": discrepancy,

            # Number of images
            "num_images": len(dicoms_for_this_study)
        }

    # Convert to DataFrame and save to CSV
    study_records = list(study_dict.values())
    df_out = pd.DataFrame(study_records)
    df_out.to_csv(out_csv, index=False)

    print(f"Created {len(study_records)} study-level records in {out_csv}.")
    print("Counts of missing sections:")
    print("  Both Findings & Impression missing:", no_imp_find)
    print("  Impression missing:", no_imp)
    print("  Findings missing:", no_find)
    print("  Last paragraph missing:", no_last)

    # Print token statistics
    print("\nToken statistics:")
    for col in [
        'findings_tokens_gpt2',
        'impression_tokens_gpt2',
        'last_paragraph_tokens_gpt2',
        'all_text_tokens_gpt2'
    ]:
        stats = df_out[col].describe()
        print(f"\n{col}:")
        print(f"  count: {int(stats['count'])}")
        print(f"  mean:  {stats['mean']:.1f}")
        print(f"  std:   {stats['std']:.1f}")
        print(f"  min:   {stats['min']:.0f}")
        print(f"  25%:   {stats['25%']:.0f}")
        print(f"  50%:   {stats['50%']:.0f}")
        print(f"  75%:   {stats['75%']:.0f}")
        print(f"  max:   {stats['max']:.0f}")


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    create_mimic_study_level_metadata_csv(
        image_root="/example/mimic-cxr-jpg/files",
        text_root="/example/mimic-cxr-reports/files",
        split_csv="/example/mimic-cxr-2.0.0-split.csv",
        chexpert_csv="/example/mimic-cxr-2.0.0-chexpert.csv",
        meta_csv="/example/mimic-cxr-2.0.0-metadata.csv",
        out_csv="/example/mimic_metadata_bert_base_macbook.csv",
        chexbert_ckpt="/example/chexbert.pth",
        device="mps"  # or "cpu"
    )
