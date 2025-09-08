# app.py
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(layout="wide", page_title="FHIR Immunization Validation")
st.title("FHIR Immunization Validation Dashboard")

st.markdown(
    "This dashboard shows how a dataset with issues can be checked using **FHIR-style rules** to "
    "identify errors and provide suggestions. Short, visual, and interactive. 🚀"
)

with st.expander("What is FHIR? (one-liner)"):
    st.caption(
        "FHIR (Fast Healthcare Interoperability Resources) is a global standard that makes health data consistent "
        "and shareable. Here we apply its *Immunization* concepts to check vaccine records."
    )

# ----------------------------
# 1) Upload or Generate Dataset
# ----------------------------
st.subheader("1️⃣ Data — Upload or Use Synthetic")
uploaded = st.file_uploader("Upload your immunization CSV (optional)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ File uploaded.")
else:
    st.info("No file uploaded — using a synthetic dataset with a few deliberate issues.")
    patients = ["P001", "P002", "P003", "P004", "P005"]
    names = ["Alice", "Bob", "Charlie", "Diana", "Ethan"]
    vaccines = ["MMR", "DTaP", "Polio", "HepB", "MenB", "IPV", "MMR Extra", ""]  # include some invalids
    base = datetime(2023, 1, 1)
    dates = [base + timedelta(days=random.randint(0, 700)) for _ in range(80)]

    df = pd.DataFrame({
        "Id": range(1, 81),
        "PATIENT": [random.choice(patients) for _ in range(80)],
        "PATIENT_NAME": [random.choice(names) for _ in range(80)],
        "DESCRIPTION": [random.choice(vaccines) for _ in range(80)],
        "VACCINE DATE AND TIME": dates,
        "BIRTH_DATE": [datetime(2022, 1, 1) + timedelta(days=random.randint(0, 500)) for _ in range(80)],
        "GENDER": [random.choice(["M", "F", ""]) for _ in range(80)],
    })

# Safe conversions
def _parse_date_series(s):
    if s is None or s.empty:
        return pd.Series(pd.NaT, index=df.index)
    # Try flexible parsing (handles 20/1/2024 and 2024-01-20 etc.)
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)

df["occurrenceDateTime"] = _parse_date_series(df.get("VACCINE DATE AND TIME"))
df["birthDate"] = _parse_date_series(df.get("BIRTH_DATE"))

st.dataframe(df.head(10))
st.caption("📋 Preview of raw data (first 10 rows).")

# ----------------------------
# 2) Pre-FHIR Validation (flags & suggestions)
# ----------------------------
st.subheader("2️⃣ Pre-FHIR Validation — Row-level Flags & Suggestions")

# Simple demo value set (you can wire to SNOMED/official value sets later)
FHIR_VACCINES = ["MMR", "DTaP", "Polio", "HepB", "MenB", "IPV"]

def validate_row(row, df_full):
    flags, suggestions = [], []

    # Patient ID
    patient = row.get("PATIENT")
    if not patient or pd.isna(patient) or str(patient).strip() == "":
        flags.append("Missing Patient ID")
        suggestions.append("Add a valid Patient identifier (Patient.identifier).")

    # Vaccine
    desc = row.get("DESCRIPTION")
    if not desc or pd.isna(desc) or str(desc).strip() == "":
        flags.append("Missing Vaccine Code/Text")
        suggestions.append("Provide Immunization.vaccineCode (code/display).")
    elif str(desc) not in FHIR_VACCINES:
        flags.append(f"Non-standard Vaccine: {desc}")
        suggestions.append(f"Map to a valid value set (e.g., {FHIR_VACCINES}).")

    # Occurrence date
    occ = row.get("occurrenceDateTime")
    if pd.isna(occ):
        flags.append("Missing/Invalid occurrenceDateTime")
        suggestions.append("Set Immunization.occurrence[x] to a valid past datetime.")
    elif occ > datetime.now():
        flags.append("Future occurrenceDateTime")
        suggestions.append("Correct to a past or current datetime.")

    # Birth vs occurrence sanity
    bd = row.get("birthDate")
    if (not pd.isna(bd)) and (not pd.isna(occ)) and (occ < bd):
        flags.append("occurrenceDateTime before birthDate")
        suggestions.append("Correct occurrenceDateTime to be on/after birthDate.")

    # Duplicate heuristic
    dupe = df_full[
        (df_full["PATIENT"] == row.get("PATIENT")) &
        (df_full["occurrenceDateTime"] == row.get("occurrenceDateTime")) &
        (df_full["DESCRIPTION"] == row.get("DESCRIPTION"))
    ]
    if len(dupe) > 1:
        flags.append("Possible Duplicate Record")
        suggestions.append("Deduplicate identical (Patient, Vaccine, Date) entries.")

    # Optional: gender presence
    gender = row.get("GENDER")
    if gender is None or str(gender).strip() == "":
        flags.append("Missing Gender")
        suggestions.append("Set Patient.gender (male | female | other | unknown).")

    return "; ".join(flags), "; ".join(suggestions)

df[["FHIR Error Flags", "FHIR Suggestions"]] = df.apply(lambda r: validate_row(r, df), axis=1, result_type="expand")

# Small helper columns for UX
df["⚠️ Any error?"] = df["FHIR Error Flags"].apply(lambda s: "⚠️" if isinstance(s, str) and s.strip() else "—")
df["Error Types (list)"] = df["FHIR Error Flags"].apply(lambda s: [e.strip() for e in str(s).split(";") if e.strip()])

# Interactive filters
f1, f2, f3 = st.columns([1, 1, 1.2])
with f1:
    sel_pat = st.multiselect("Filter: Patient", sorted(df["PATIENT"].dropna().unique().tolist()))
with f2:
    sel_vax = st.multiselect("Filter: Vaccine", sorted(df["DESCRIPTION"].fillna("").unique().tolist()))
with f3:
    all_errs = sorted({e for lst in df["Error Types (list)"] for e in lst})
    sel_err = st.multiselect("Filter: Error Type", all_errs)

pre_mask = pd.Series([True] * len(df))
if sel_pat: pre_mask &= df["PATIENT"].isin(sel_pat)
if sel_vax: pre_mask &= df["DESCRIPTION"].isin(sel_vax)
if sel_err: pre_mask &= df["Error Types (list)"].apply(lambda lst: any(e in lst for e in sel_err))

pre_view = df.loc[pre_mask, ["PATIENT", "PATIENT_NAME", "DESCRIPTION", "occurrenceDateTime", "⚠️ Any error?", "FHIR Error Flags", "FHIR Suggestions"]]
st.dataframe(pre_view)
st.caption("📋 Errors & suggestions BEFORE applying fixes. ‘⚠️’ indicates the row has one or more issues.")

# Visual: Error counts by type (Pre)
pre_errors_long = pd.DataFrame(
    [{"Error": e} for s in df["FHIR Error Flags"] for e in str(s).split(";") if e.strip()]
)
if not pre_errors_long.empty:
    fig_pre_err = px.bar(
        pre_errors_long.value_counts("Error").reset_index(name="Count"),
        x="Count", y="Error", orientation="h", title="Pre-FHIR: Error Types"
    )
    fig_pre_err.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_pre_err, use_container_width=True)
    st.caption("📊 Count of each error type before FHIR-style transformation.")

# ----------------------------
# 3) Simple FHIR-Style Transformation
# ----------------------------
st.subheader("3️⃣ Transformation — Standardize Vaccine & Fix Dates (visual only)")

def transform_row(row):
    # Vaccine standardization
    desc = row["DESCRIPTION"]
    row["DESCRIPTION_corrected"] = desc if (isinstance(desc, str) and desc in FHIR_VACCINES) else random.choice(FHIR_VACCINES)

    # Occurrence fixes
    occ = row["occurrenceDateTime"]
    bd = row["birthDate"]

    if pd.isna(occ):
        # if missing, synthesize a plausible date after birth if we have it, else default to now-1 day
        occ_new = (bd + timedelta(days=random.randint(0, 365))) if not pd.isna(bd) else (datetime.now() - timedelta(days=1))
    else:
        occ_new = occ

    if not pd.isna(bd) and occ_new < bd:
        occ_new = bd + timedelta(days=random.randint(0, 365))
    if occ_new > datetime.now():
        occ_new = datetime.now()

    row["occurrenceDateTime_corrected"] = occ_new
    return row

df = df.apply(transform_row, axis=1)

# Re-validate after transformation (use corrected fields)
df[["Post-FHIR Error Flags", "Post-FHIR Suggestions"]] = df.apply(
    lambda r: validate_row(
        {
            **r,
            "DESCRIPTION": r["DESCRIPTION_corrected"],
            "occurrenceDateTime": r["occurrenceDateTime_corrected"]
        },
        df.assign(
            DESCRIPTION=lambda x: x["DESCRIPTION_corrected"],
            occurrenceDateTime=lambda x: x["occurrenceDateTime_corrected"]
        )
    ),
    axis=1,
    result_type="expand"
)

df["⚠️ Any error? (post)"] = df["Post-FHIR Error Flags"].apply(lambda s: "⚠️" if isinstance(s, str) and s.strip() else "—")

# Visual: Sankey showing error flow
total_records = int(len(df))
pre_rows_with_error = int((df["FHIR Error Flags"].str.len() > 0).sum())
post_rows_with_error = int((df["Post-FHIR Error Flags"].str.len() > 0).sum())

labels = [
    "Raw Records",
    "Rows with Pre-FHIR Errors",
    "Rows without Pre-FHIR Errors",
    "Transformed (FHIR-style)",
    "Rows with Post-FHIR Errors",
    "Rows Clean (Post-FHIR)"
]
idx = {k: i for i, k in enumerate(labels)}

flow1 = pre_rows_with_error
flow2 = total_records - pre_rows_with_error
flow4 = post_rows_with_error
flow5 = total_records - post_rows_with_error

sankey = go.Figure(data=[go.Sankey(
    node=dict(pad=18, thickness=18, line=dict(width=0.5), label=labels),
    link=dict(
        source=[
            idx["Raw Records"], idx["Raw Records"],
            idx["Rows with Pre-FHIR Errors"], idx["Rows without Pre-FHIR Errors"],
            idx["Transformed (FHIR-style)"], idx["Transformed (FHIR-style)"]
        ],
        target=[
            idx["Rows with Pre-FHIR Errors"], idx["Rows without Pre-FHIR Errors"],
            idx["Transformed (FHIR-style)"], idx["Transformed (FHIR-style)"],
            idx["Rows with Post-FHIR Errors"], idx["Rows Clean (Post-FHIR)"]
        ],
        value=[int(flow1), int(flow2), int(flow1), int(flow2), int(flow4), int(flow5)]
    )
)])
sankey.update_layout(title_text="Flow: Raw → Pre-FHIR → Transform → Post-FHIR", font_size=12)
st.plotly_chart(sankey, use_container_width=True)
st.caption("🔀 Visual flow of rows with/without errors as they move through the process.")

# ----------------------------
# 4) Post-FHIR Validation — Remaining Issues
# ----------------------------
st.subheader("4️⃣ Post-FHIR — Remaining Issues (Filtered)")
# Post filters
pf1, pf2, pf3 = st.columns([1, 1, 1.2])
with pf1:
    sel_pat_post = st.multiselect("Filter: Patient (post)", sorted(df["PATIENT"].dropna().unique().tolist()))
with pf2:
    sel_vax_post = st.multiselect("Filter: Vaccine (post)", sorted(df["DESCRIPTION_corrected"].fillna("").unique().tolist()))
with pf3:
    all_errs_post = sorted({e for s in df["Post-FHIR Error Flags"] for e in str(s).split(";") if e.strip()})
    sel_err_post = st.multiselect("Filter: Error Type (post)", all_errs_post)

post_mask = pd.Series([True] * len(df))
if sel_pat_post: post_mask &= df["PATIENT"].isin(sel_pat_post)
if sel_vax_post: post_mask &= df["DESCRIPTION_corrected"].isin(sel_vax_post)
if sel_err_post:
    post_mask &= df["Post-FHIR Error Flags"].apply(
        lambda s: any(e in [x.strip() for x in str(s).split(";")] for e in sel_err_post)
    )

post_view = df.loc[post_mask, [
    "PATIENT", "PATIENT_NAME", "DESCRIPTION_corrected", "occurrenceDateTime_corrected",
    "⚠️ Any error? (post)", "Post-FHIR Error Flags", "Post-FHIR Suggestions"
]].rename(columns={
    "DESCRIPTION_corrected": "DESCRIPTION",
    "occurrenceDateTime_corrected": "occurrenceDateTime"
})
st.dataframe(post_view)
st.caption("📋 Errors & suggestions AFTER transformation (should be fewer).")

# ----------------------------
# 5) KPIs / Summary metrics
# ----------------------------
st.subheader("5️⃣ Summary Metrics")

total_error_instances_pre = int(
    df["FHIR Error Flags"].apply(lambda s: len([e for e in str(s).split(";") if e.strip()])).sum()
)
total_error_instances_post = int(
    df["Post-FHIR Error Flags"].apply(lambda s: len([e for e in str(s).split(";") if e.strip()])).sum()
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records", int(total_records))
c2.metric("Rows with Errors (Pre)", int(pre_rows_with_error))
c3.metric("Rows with Errors (Post)", int(post_rows_with_error), delta=int(post_rows_with_error - pre_rows_with_error))
c4.metric("Error Instances (Pre)", int(total_error_instances_pre))
c5.metric("Error Instances (Post)", int(total_error_instances_post), delta=int(total_error_instances_post - total_error_instances_pre))

st.caption("ℹ️ *Rows with errors* counts rows containing ≥1 error. *Error instances* counts each individual error across all rows.")

# ----------------------------
# 6) Visualizations
# ----------------------------
st.subheader("6️⃣ Visualizations")

# 6a) Error Types: Pre vs Post (horizontal bar)
comparison_df = pd.DataFrame(
    [{"Error": e.strip(), "Stage": "Pre-FHIR"} for s in df["FHIR Error Flags"] for e in str(s).split(";") if e.strip()] +
    [{"Error": e.strip(), "Stage": "Post-FHIR"} for s in df["Post-FHIR Error Flags"] for e in str(s).split(";") if e.strip()]
)
if not comparison_df.empty:
    fig1 = px.bar(
        comparison_df.groupby(["Error", "Stage"]).size().reset_index(name="Count"),
        x="Count", y="Error", color="Stage", orientation="h",
        title="Error Types — Pre vs Post"
    )
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("🧭 Compare counts of each error type before and after transformation.")

# 6b) Per-Patient Errors (grouped bar)
per_patient = pd.DataFrame(
    [{"Patient": r["PATIENT"], "Stage": "Pre-FHIR",
      "Errors": len([e for e in str(r["FHIR Error Flags"]).split(";") if e.strip()])} for _, r in df.iterrows()] +
    [{"Patient": r["PATIENT"], "Stage": "Post-FHIR",
      "Errors": len([e for e in str(r["Post-FHIR Error Flags"]).split(";") if e.strip()])} for _, r in df.iterrows()]
)
if not per_patient.empty:
    fig2 = px.bar(per_patient, x="Patient", y="Errors", color="Stage", barmode="group",
                  title="Per-Patient Errors — Pre vs Post")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("👤 Patient-level error distribution before and after transformation.")

# 6c) Errors Fixed vs Remaining per Error Type (stacked)
fixed_rows = []
for _, r in df.iterrows():
    pre_es = [e.strip() for e in str(r["FHIR Error Flags"]).split(";") if e.strip()]
    post_es = [e.strip() for e in str(r["Post-FHIR Error Flags"]).split(";") if e.strip()]
    for e in pre_es:
        fixed_rows.append({"Error": e, "Fixed": 1 if e not in post_es else 0})
fixed_df = pd.DataFrame(fixed_rows)
if not fixed_df.empty:
    fx = fixed_df.groupby("Error")["Fixed"].sum().reset_index()
    total_pre_by_err = fixed_df.groupby("Error").size().reset_index(name="Total")
    fx = fx.merge(total_pre_by_err, on="Error")
    fx["Remaining"] = fx["Total"] - fx["Fixed"]

    fig3 = px.bar(
        fx, x="Error", y=["Fixed", "Remaining"],
        title="Errors Fixed vs Remaining (by Error Type)", text_auto=True
    )
    fig3.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🛠️ Which error types were fixed most effectively?")

# ----------------------------
# 7) FHIR-Ready (Concise View)
# ----------------------------
st.subheader("7️⃣ FHIR-Ready (Concise View)")
fhir_ready = df[[
    "PATIENT", "PATIENT_NAME", "DESCRIPTION_corrected", "occurrenceDateTime_corrected",
    "birthDate", "GENDER"
]].rename(columns={
    "DESCRIPTION_corrected": "FHIR.vaccineCode.text",
    "occurrenceDateTime_corrected": "FHIR.occurrenceDateTime",
    "birthDate": "FHIR.patient.birthDate",
    "GENDER": "FHIR.patient.gender"
})
st.dataframe(fhir_ready.head(15))
st.caption("✅ Minimal fields aligned to FHIR Immunization/Patient (for illustration; no JSON export here).")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "This demo uses simple FHIR-style checks. For production, connect to official value sets (e.g., SNOMED) "
    "and a FHIR validator for full compliance."
)
