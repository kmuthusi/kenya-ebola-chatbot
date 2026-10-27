# WHO Ebola and filovirus reference documents

Downloaded from official WHO publication pages on 27 October 2026. These are
reference copies; the chatbot currently indexes only the Kenya guideline in the
repository root.

| Local file | WHO source | Type |
| --- | --- | --- |
| [clinical-management-filovirus-2026.pdf](clinical-management-filovirus-2026.pdf) | [Clinical management of filovirus disease](https://www.who.int/publications/i/item/B09774) | Guideline |
| [clinical-management-filovirus-executive-summary-2026.pdf](clinical-management-filovirus-executive-summary-2026.pdf) | [Executive summary](https://www.who.int/publications/i/item/B09786) | Guideline summary |
| [optimized-supportive-care-ebola-2019.pdf](optimized-supportive-care-ebola-2019.pdf) | [Optimized supportive care](https://www.who.int/publications-detail-redirect/9789241515894) | Clinical procedures |
| [therapeutics-ebola-2022.pdf](therapeutics-ebola-2022.pdf) | [Therapeutics for Ebola virus disease](https://www.who.int/publications/i/item/9789240055742) | Guideline |
| [ipc-ebola-marburg-2026.pdf](ipc-ebola-marburg-2026.pdf) | [Infection prevention and control](https://www.who.int/publications/i/item/9789240111332) | Guideline |
| [diagnostic-testing-ebola-marburg-2026.pdf](diagnostic-testing-ebola-marburg-2026.pdf) | [Diagnostic testing](https://www.who.int/publications/i/item/B09834) | Interim guidance |
| [mab114-poster-2022.pdf](mab114-poster-2022.pdf) | [mAb114 poster](https://www.who.int/publications/i/item/WHO-EVD-therapeutics-mAb114-Poster_A-2022.1) | Treatment toolkit |
| [regn-eb3-poster-2022.pdf](regn-eb3-poster-2022.pdf) | [REGN-EB3 poster](https://www.who.int/publications/i/item/WHO-EVD-therapeutics-REGN-EB3-Poster_A-2022.1) | Treatment toolkit |
| [end-of-ebola-outbreak-criteria-2020.pdf](end-of-ebola-outbreak-criteria-2020.pdf) | [End-of-outbreak criteria](https://www.who.int/publications/m/item/who-recommended-criteria-for-declaring-the-end-of-the-ebola-virus-disease-outbreak) | Technical note |

The [WHO Ebola disease fact sheet](https://www.who.int/news-room/fact-sheets/detail/ebola-disease)
is a web page rather than a downloadable guideline. It remains linked here for
reference. The CDC, JAMA and NCBI links in the request are external references,
not WHO guidelines.

Run `python scripts/download_who_guidelines.py` from the repository root to
download missing PDFs again. Existing files with a PDF signature are skipped.
