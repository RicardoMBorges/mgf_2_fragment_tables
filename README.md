# mgf_2_fragment_tables


# MGF → Fragment Tables (with mzML support)

A Streamlit app to parse **mass spectrometry files** (`.mgf` and `.mzML`), 
summarize the **top fragment ions** per spectrum, and provide interactive visualization 
and CSV export.



## Features

- **Upload or point to files**: Works with `.mgf` and `.mzML`.
- **Interactive filtering**: Choose top-N fragments, set minimum relative intensity (%).
- **Table view**: See batch, scan number, precursor mass, and fragment strings.
- **Fragment preview**: Bar chart of relative intensities with hover.
- **Export**: Download the summary table as CSV.
- **Search**: Filter rows by substring across all fields.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YourUsername/mgf_2_fragment_tables.git
cd mgf_2_fragment_tables

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # on Linux/macOS
venv\Scripts\activate      # on Windows

# Install requirements
pip install -r requirements.txt
````

Dependencies include:

* `streamlit`
* `pyteomics`
* `numpy`
* `pandas`
* `plotly`
* `lxml` (for mzML parsing)

---

## Run the app locally

```bash
streamlit run app.py
```

Then open the link shown in the terminal, e.g. [http://localhost:8501](http://localhost:8501).

---

## Online version

The app can also be deployed on **Streamlit Cloud**.
Just point to `app.py` as the **Main file path** in your app settings.

---

## Tutorial

### Step 1 — Upload files

* Go to the sidebar and select **Upload .mgf/.mzML file(s)**.
* Drag & drop one or more `.mgf` or `.mzML` files.
* Click **Build table from uploads**.

Alternatively, choose **Use local path** to point to a folder or single file
(if running locally).

---

### Step 2 — Configure fragment extraction

* **Top N fragments to keep**: keep only the most intense N fragment ions (default 6).
* **Min relative intensity (%)**: discard peaks below this threshold.

---

### Step 3 — Explore results

* The summary table shows:

  * `batch`: filename
  * `scans`: scan identifier
  * `scan_number`: numeric scan number (if available)
  * `precursor_mass`: precursor m/z
  * `n_fragments`: number of fragment ions kept
  * `fragments`: semicolon-separated `"mz:rel%"`

* Use the search box to filter rows by keyword (batch name, scan number, m/z…).

---

### Step 4 — Inspect a spectrum

* Choose a row in the dropdown to preview its fragments.
* A **bar chart** shows relative intensities vs m/z.
* The right panel shows raw values for the row.

---

### Step 5 — Export

* Click **Download table as CSV** to save the results.

---

## Example output

| batch   | scans    | scan_number | precursor_mass | n_fragments | fragments             |
| ------- | -------- | ----------- | -------------- | ----------- | --------------------- |
| Sample1 | scan=123 | 123         | 456.789        | 6           | 100.1:80; 150.2:65; … |

---

## 🛠️ Notes

* Only **MS/MS (MS2+) scans** from `.mzML` are included.
* Precursor m/z is extracted from `precursorList.selectedIonList`.
* This tool is intended for **survey/summary inspection**, not quantitative analysis.

---

## 👩‍💻 Authors

Developed by **Ricardo M. Borges** and collaborators at **LAABio-IPPN-UFRJ**.

Contact: `ricardo_mborges@ufrj.br`

---

Would you like me to also include **screenshots placeholders** (Markdown image tags) so the README looks more like a real tutorial when you add images later?
```
