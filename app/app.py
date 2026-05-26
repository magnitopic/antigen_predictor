"""
app.py — Clasificador de Antigenicidad de Proteínas
Proyecto educativo de Machine Learning aplicado a bioinformática.
"""

import streamlit as st
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
from io import StringIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Antigenicidad",
    page_icon="🧬",
    layout="wide"
)

# ── Cargar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Ruta compatible con ejecución local (app/) y Streamlit Community Cloud (raíz)
    from pathlib import Path
    _here = Path(__file__).parent          # carpeta donde está app.py
    _root = _here.parent                   # raíz del repositorio
    _model_path = _root / "models" / "model.pkl"
    if not _model_path.exists():
        _model_path = _here / "model.pkl"  # fallback para compatibilidad
    return joblib.load(_model_path)

bundle = load_model()
model        = bundle["model"]
feature_cols = bundle["feature_cols"]
amino_acids  = bundle["amino_acids"]
label_map    = bundle["label_map"]
auc_cv       = bundle["auc_cv"]
n_train      = bundle["n_train"]

# ── Funciones ────────────────────────────────────────────────────────────────
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq: str) -> str:
    """Elimina aminoácidos no estándar y espacios."""
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper().replace(" ", "").replace("\n", ""))


def compute_features(seq: str) -> dict | None:
    """Calcula las 24 features fisicoquímicas de una secuencia."""
    seq = clean_sequence(seq)
    if len(seq) < 10:
        return None
    try:
        analysis = ProteinAnalysis(seq)
        features = {
            "length":            len(seq),
            "molecular_weight":  analysis.molecular_weight(),
            "isoelectric_point": analysis.isoelectric_point(),
            "gravy":             analysis.gravy(),
        }
        aa_comp = analysis.amino_acids_percent
        for aa in amino_acids:
            features[f"aa_{aa}"] = aa_comp.get(aa, 0.0)
        return features
    except Exception:
        return None


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """
    Parsea texto en formato FASTA.
    Devuelve lista de (nombre, secuencia).
    """
    proteins = []
    current_name = None
    current_seq  = []

    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_name is not None:
                proteins.append((current_name, "".join(current_seq)))
            current_name = line[1:].split()[0]  # primer token tras >
            current_seq  = []
        elif line:
            current_seq.append(line)

    if current_name is not None:
        proteins.append((current_name, "".join(current_seq)))

    return proteins


def predict_proteins(proteins: list[tuple[str, str]]) -> list[dict]:
    """Calcula features y predice para una lista de (nombre, secuencia)."""
    results = []
    for name, seq in proteins:
        feats = compute_features(seq)
        if feats is None:
            results.append({
                "Proteína": name,
                "Longitud": len(clean_sequence(seq)),
                "Score antigenicidad": None,
                "Predicción": "⚠️ Secuencia inválida o muy corta",
                "MW (Da)": None,
                "pI": None,
                "GRAVY": None,
            })
            continue

        X = np.array([[feats[col] for col in feature_cols]])
        score = model.predict_proba(X)[0][1]
        label = "🟢 Antigénica" if score >= 0.5 else "🔴 No antigénica"

        results.append({
            "Proteína":            name,
            "Longitud":            feats["length"],
            "Score antigenicidad": round(score, 4),
            "Predicción":          label,
            "MW (Da)":             round(feats["molecular_weight"], 1),
            "pI":                  round(feats["isoelectric_point"], 2),
            "GRAVY":               round(feats["gravy"], 4),
        })

    return sorted(
        [r for r in results if r["Score antigenicidad"] is not None],
        key=lambda x: x["Score antigenicidad"],
        reverse=True
    ) + [r for r in results if r["Score antigenicidad"] is None]


def plot_scores(results: list[dict]) -> plt.Figure:
    """Genera gráfico de barras horizontal con los scores."""
    valid = [r for r in results if r["Score antigenicidad"] is not None]
    if not valid:
        return None

    names  = [r["Proteína"][:40] for r in valid]
    scores = [r["Score antigenicidad"] for r in valid]
    colors = ["#5cb85c" if s >= 0.5 else "#d9534f" for s in scores]

    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.5 + 1)))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, label="Umbral (0.5)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score de antigenicidad", fontsize=11)
    ax.set_title("Predicción de antigenicidad por proteína", fontsize=13)
    ax.legend(fontsize=9)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center", fontsize=9
        )

    plt.tight_layout()
    return fig


# ── Interfaz ─────────────────────────────────────────────────────────────────
st.title("🧬 Clasificador de Antigenicidad de Proteínas")
st.markdown(
    "Herramienta educativa de Machine Learning para priorizar candidatos antigénicos "
    "a partir de secuencias de proteínas en formato FASTA."
)

# Sidebar con info del modelo
with st.sidebar:
    st.header("ℹ️ Información del modelo")
    st.metric("AUC-ROC (CV k=5)", f"{auc_cv:.3f}")
    st.metric("Proteínas de entrenamiento", f"{n_train:,}")
    st.markdown("---")
    st.markdown("**Modelo:** Random Forest (200 árboles)")
    st.markdown("**Features:** 24 (4 fisicoquímicas + 20 composición aa)")
    st.markdown("**Patógenos de entrenamiento:** SARS-CoV-2, Influenza A")
    st.markdown("**Fuente de datos:** IEDB (tcell + bcell)")
    st.markdown("---")
    st.markdown(
        "⚠️ **Limitaciones**\n\n"
        "- Solo secuencia de aminoácidos, sin estructura 3D\n"
        "- Entrenado en 2 patógenos\n"
        "- No es un predictor clínico\n"
        "- Score alto no garantiza buen antígeno vacunal"
    )

# Área principal
st.markdown("### Introduce las secuencias proteicas")

_EXAMPLE_FASTA = (
    ">Spike_glycoprotein\n"
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT\n"
    ">Nucleoprotein\n"
    "MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQGVPINTNSSPDDQIGYYRRATRRIRGGDGKMKDLSPRWYFYYLGTGPEAGLPYGANKDGIIWVATEGALNTPKDHIGTRNPANNAAIVLQLPQGTTLPKGFYAEGSRGGSQASSRSSSRSRNSSRNSTPGSSRGTSPARMAGNGGDAALALLLLDRLNQLESKMSGKGQQQQGQTVTKKSAAEASKKPRQKRTATKAYNNTQGVELKDDNPQPGLPNQLRNLSDSRFQGFNRTQIMSSNPGQLKPIRESADVSQNKTPKVNRSAANKTAAANDSSFSSVQSSDFPNSLPNLPQPNSPQGNQDESSKNQNNPEKDLSPEQPPHSSQAFEAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVP\n"
    ">Membrane_protein\n"
    "MADSNGTITVEELKKLLEQWNLVIGFLFLTWICLLQFAYANRNRFLYIIKLIFLWLLWPVTLACFVLAAVYRINWITGGIAIAMACLVGLMWLSYFIASFRLFARTRSMWSFNPETNILLNVPLHGTILTRPLLESELVIGAVINSYVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQ\n"
    ">Envelope_protein\n"
    "MYSFVSEETGTLIVNSVLLFLAFVVFLLVTLAILTALRLCAYCCNIVNVSLVKPSFYVYSRVKNLNSSRVPDLLV\n"
)

# Si el botón de ejemplo fue pulsado en el rerun anterior, cargar las secuencias
# ANTES de instanciar el widget (Streamlit no permite modificar st.session_state[key]
# después de que el widget con ese key haya sido creado).
if st.session_state.pop("_load_example", False):
    st.session_state["fasta_input"] = _EXAMPLE_FASTA

col1, col2 = st.columns([3, 1])

with col1:
    fasta_input = st.text_area(
        "Pega aquí tu archivo FASTA:",
        height=250,
        placeholder=(
            ">Spike_glycoprotein\n"
            "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHV\n"
            ">Nucleoprotein\n"
            "MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQ..."
        ),
        help="Formato FASTA estándar. Una o más proteínas.",
        key="fasta_input"
    )

with col2:
    st.markdown("**O sube un archivo .fasta**")
    uploaded_file = st.file_uploader(
        "Archivo FASTA",
        type=["fasta", "fa", "txt"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        fasta_input = uploaded_file.read().decode("utf-8")
        st.success(f"Archivo cargado: {uploaded_file.name}")

    st.markdown("---")
    st.markdown("**Ejemplo rápido**")
    if st.button("Cargar proteínas SARS-CoV-2", use_container_width=True):
        st.session_state["_load_example"] = True
        st.rerun()

# Botón de predicción
if st.button("🔬 Predecir antigenicidad", type="primary", use_container_width=True):
    if not fasta_input or not fasta_input.strip():
        st.error("Por favor introduce al menos una secuencia en formato FASTA.")
    else:
        proteins = parse_fasta(fasta_input)
        if not proteins:
            st.error("No se encontraron secuencias válidas. Verifica el formato FASTA.")
        else:
            with st.spinner(f"Analizando {len(proteins)} proteína(s)..."):
                results = predict_proteins(proteins)

            st.markdown("---")
            st.markdown(f"### Resultados — {len(proteins)} proteína(s) analizadas")

            # Gráfico
            fig = plot_scores(results)
            if fig:
                st.pyplot(fig)
                plt.close()

            # Tabla
            import pandas as pd
            df_results = pd.DataFrame(results)
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score antigenicidad": st.column_config.ProgressColumn(
                        "Score antigenicidad",
                        min_value=0, max_value=1,
                        format="%.4f"
                    )
                }
            )

            # Descarga CSV
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar resultados (CSV)",
                data=csv,
                file_name="antigenicidad_resultados.csv",
                mime="text/csv"
            )

            # Nota metodológica
            with st.expander("ℹ️ Cómo interpretar los resultados"):
                st.markdown(
                    "**Score de antigenicidad:** probabilidad estimada por el modelo "
                    "de que la proteína sea reconocida por el sistema inmune humano. "
                    "Varía entre 0 (no antigénica) y 1 (muy probablemente antigénica). "
                    "El umbral de decisión es **0.5**.\n\n"
                    "**Esta herramienta es un filtro orientativo**, no un predictor clínico. "
                    "Un score alto prioriza la proteína para estudio experimental, "
                    "pero no garantiza que sea un buen antígeno vacunal. "
                    "La validación experimental es siempre necesaria.\n\n"
                    "**Limitaciones principales:**\n"
                    "- Solo usa información de secuencia (sin estructura 3D)\n"
                    "- Entrenado con datos de SARS-CoV-2 e Influenza A\n"
                    "- La generalización a otros patógenos es desconocida"
                )
