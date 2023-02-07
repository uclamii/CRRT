from argparse import Namespace
import difflib
import pickle
import sys
from os import getcwd
from os.path import join
from pandas import DataFrame
from fuzzywuzzy import fuzz

sys.path.insert(0, join(getcwd(), "module_code"))

from cli_utils import load_cli_args, init_cli_args
from data.utils import FILE_NAMES, read_files_and_combine


def create_and_serialize_medication_mapping_dict(
    args: Namespace, only_cedars: set, all_ucla: set
) -> None:

    exceptions = {
        "ANTIDEPRESSANT - SEROTONIN-2 ANTAGONIST-REUPTAKE INHIBITORS (SARIS)",
        "ANTIDEPRESSANT-NOREPINEPHRINE AND DOPAMINE REUPTAKE INHIBITORS (NDRIS)",
        "BLOOD CELL AND PLATELET DISORDER TX-SPLEEN TYROSINE KINASE INHIBITORS",
        "HYPERURICEMIA THERAPY - XANTHINE OXIDASE INHIBITORS",
        "NARCOLEPSY THERAPY AGENTS - NON-SYMPATHOMIMETIC",
        "URINARY RETENTION THERAPY - PARASYMPATHOMIMETIC AGENTS",
        "CALCIMIMETIC, PARATHYROID CALCIUM RECEPTOR SENSITIVITY ENHANCER",
    }

    manual_mapping = {
        "CALCIUM CHANNEL BLOCKERS - DIHYDROPYRIDINES - CEREBROVASCULAR SPECIFIC": "CALCIUM CHANNEL BLOCKERS",
        "CALCIUM CHANNEL BLOCKERS - BENZOTHIAZEPINES": "CALCIUM CHANNEL BLOCKERS",
        "BETA BLOCKERS NON-CARDIAC SELECTIVE": "BETA BLOCKERS NON-SELECTIVE",
        "ASTHMA THERAPY - LEUKOTRIENE RECEPTOR ANTAGONISTS": "LEUKOTRIENE MODULATORS",
        "CMV ANTIVIRAL AGENT - INORGANIC PYROPHOSPHATE ANALOGS": "CMV AGENTS",
        "OVERACTIVE BLADDER AGENTS - BETA -3 ADRENERGIC RECEPTOR AGONIST": "URINARY ANTISPASMODICS - BETA-3 ADRENERGIC AGONISTS 61",
        "ANTIPARKINSON - DOPAMINERG-PERIPHERAL DOPA-DECARBOXYLASE INHIBIT COMB": "ANTIPARKINSON DOPAMINERGICS",
        "CONTRAST MEDIA - BARIUM": "RADIOGRAPHIC CONTRAST MEDIA",
        "ANTIPSYCHOTIC - ATYPICAL DOPAMINE-SEROTONIN ANTAG- BENZISOTHIAZOLONES": "BENZODIAZEPINE ANTAGONISTS",
        "THROMBOLYTIC - TISSUE PLASMINOGEN ACTIVATORS": "THROMBOLYTIC ENZYMES",
        "ANTICONVULSANT - BARBITURATES AND DERIVATIVES": "ANTICONVULSANTS - MISC.",
        "ANTIHISTAMINES - 2ND GENERATION": "ANTIHISTAMINES - NON-SEDATING",
        "CONTRAST MEDIA - MAGNETIC RESONANCE GADOLINIUM COMPLEXES": "RADIOGRAPHIC CONTRAST MEDIA",
        "ANTIDEPRESSANT-TRICYCLICS AND RELATED (NON-SELECT REUPTAKE INHIBITORS)": "TRICYCLIC AGENTS",
        "ANTIARRHYTHMIC - CLASS IV": "ANTIARRHYTHMICS - MISC.",
        "AGENTS FOR OPIOID WITHDRAWAL, OPIOID-TYPE": "OPIOID PARTIAL AGONISTS",
        "ANTIANXIETY AGENT - ANTIHISTAMINE TYPE": "ANTIANXIETY AGENTS - MISC.",
        "ANTISEPTIC - OXIDIZING AGENTS": "ANTISEPTICS & DISINFECTANTS",
        "VACCINE VIRAL - INFLUENZA A AND B": "VIRAL VACCINES",
        "CONTRAST MEDIA - IODINATED NONIONIC": "RADIOGRAPHIC CONTRAST MEDIA",
        "DERMATOLOGICAL - PROTECTANTS": "MISC. DERMATOLOGICAL PRODUCTS",
        "DERMATOLOGICAL - ANTIBACTERIAL MIXTURES": "MISC. DERMATOLOGICAL PRODUCTS",
        "ANTIDOTE - CYANIDE POISONING": "ANTIDOTES - CHELATING AGENTS",
        "PENICILLIN ANTIBIOTIC - NATURAL": "NATURAL PENICILLINS",
        "ANTISEPTIC - CHLORINE RELEASING": "CHLORINE ANTISEPTICS",
        "DEXTROSE SOLUTIONS": "PERITONEAL DIALYSIS SOLUTIONS",
        "DERMATOLOGICAL - ENZYME COMBINATIONS OTHER": "ENZYMES",
        "ANTIANXIETY AGENT - NON-BENZODIAZEPINE": "ANTIANXIETY AGENTS - MISC.",
        "ANTICONVULSANT - CARBOXYLIC ACID DERIVATIVES": "ANTICONVULSANTS - MISC.",
        "ANTIARRHYTHMIC - CLASS IB": "ANTIARRHYTHMICS - MISC.",
        "CEPHALOSPORIN ANTIBIOTICS - 1ST GENERATION": "CEPHALOSPORINS - 1ST GENERATION",
        "LAXATIVE - BULK FORMING": "BULK LAXATIVES",
        "ASTHMA THERAPY - INHALED CORTICOSTEROIDS (GLUCOCORTICOIDS)": "GLUCOCORTICOSTEROIDS",
        "ASTHMA/COPD THERAPY - BETA ADRENERGIC-GLUCOCORTICOID COMBINATIONS": "GLUCOCORTICOSTEROIDS",
        "DERMATOLOGICAL - GLUCOCORTICOID": "GLUCOCORTICOSTEROIDS",
        "ANTINEOPLASTIC - CD20 SPECIFIC RECOMBINANT MONOCLONAL ANTIBODY AGENTS": "MONOCLONAL ANTIBODIES",
        "ANTINEOPLASTIC - PHOTOSENSITIZERS": "ANTINEOPLASTICS MISC.",
        "ACNE THERAPY TOPICAL - ANTI-INFECTIVE": "ANTI-INFECTIVE AGENTS - MISC.",
        "ANALGESIC OPIOID HYDROMORPHONE COMBINATIONS": "ANALGESIC COMBINATIONS",
        "CONTRAST MEDIA - IODINATED IONIC": "RADIOGRAPHIC CONTRAST MEDIA",
        "ANTICONVULSANT - FUNCTIONALIZED AMINO ACID": "ANTICONVULSANTS - MISC.",
        "ANTIARRHYTHMIC - CLASS II": "ANTIARRHYTHMICS - MISC.",
        "MINERALS AND ELECTROLYTES - IODINE": "ELECTROLYTE MIXTURES",
        "VITAMINS - B-1, THIAMINE AND DERIVATIVES": "B-COMPLEX VITAMINS",
        "VITAMINS - B-6, PYRIDOXINE AND DERIVATIVES": "B-COMPLEX VITAMINS",
        "VITAMINS - FOLIC ACID AND DERIVATIVES": "B-COMPLEX VITAMINS",
        "ALTERNATIVE THERAPY - UNCLASSIFIED": "ALTERNATIVE MEDICINE - U",
        "ANTIPSYCHOTIC - ATYPICAL DOPAMINE-SEROTONIN ANTAG-DIBENZODIAZEPINE DER": "ANTIPSYCHOTICS - MISC.",
        "IMMUNOSUPPRESSIVE - CALCINEURIN INHIBITORS": "IMMUNOSUPPRESSIVE AGENTS",
        "ANTIPARKINSON THERAPY - NON-ERGOT DOPAMINE AGONIST AGENTS": "ANTIPARKINSON DOPAMINERGICS",
        "HEPATITIS C - NUCLEOSIDE ANALOGS": "HEPATITIS AGENTS",
        "HEPATITIS B TREATMENT- NUCLEOSIDE ANALOGS (ANTIVIRAL)": "HEPATITIS AGENTS",
        "ANALGESIC OPIOID CODEINE COMBINATIONS": "ANALGESIC COMBINATIONS",
        "ANORECTAL - GLUCOCORTICOIDS": "GLUCOCORTICOSTEROIDS",
        "IMMUNOSUPPRESSIVE - PURINE ANALOGS": "IMMUNOSUPPRESSIVE AGENTS",
        "OPHTHALMIC - ANTI-INFLAMMATORY, NSAIDS": "NONSTEROIDAL ANTI-INFLAMMATORY AGENTS (NSAIDS)",
        "RINGER'S AND LACTATED RINGER'S SOLUTIONS": "IRRIGATION SOLUTIONS",
        "DEXTROSE AND LACTATED RINGER'S SOLUTIONS": "IRRIGATION SOLUTIONS",
        "NSAID ANALGESIC, CYCLOOXYGENASE-2 (COX-2) SELECTIVE INHIBITORS": "NONSTEROIDAL ANTI-INFLAMMATORY AGENTS (NSAIDS)",
        "CEPHALOSPORIN ANTIBIOTICS - 4TH GENERATION": "CEPHALOSPORINS - 4TH GENERATION",
        "HEMOSTATIC TOPICAL AGENTS": "HEMOSTATICS - TOPICAL",
        "OPHTHALMIC - ANTICHOLINERGICS": "OPHTHALMICS - MISC.",
        "SEDATIVE-HYPNOTIC - GABA-RECEPTOR MODULATORS": "HORMONE RECEPTOR MODULATORS",
        "MACROLIDE ANTIBIOTICS": "ANTIBIOTICS - TOPICAL",
        "ANTIBACTERIAL FOLATE ANTAGONIST - OTHER COMBINATIONS": "ANTI-INFECTIVE MISC. - COMBINATIONS",
        "CEPHALOSPORIN ANTIBIOTICS - 2ND GENERATION": "CEPHALOSPORINS - 2ND GENERATION",
        "ANTIPSYCHOTIC -ATYPICAL DOPAMINE-SEROTONIN ANTAG-DIBENZOTHIAZEPINE DER": "ANTIPSYCHOTICS - MISC.",
        "CEPHALOSPORIN ANTIBIOTICS - 3RD GENERATION": "CEPHALOSPORINS - 3RD GENERATION",
        "GALLSTONE SOLUBILIZING (LITHOLYSIS) AGENTS": "GALLSTONE SOLUBILIZING AGENTS",
        "AGENTS TO TREAT HYPOGLYCEMIA (HYPERGLYCEMICS)": "ANTIHYPERLIPIDEMICS - COMBINATIONS",
        "DERMATOLOGICAL - ANTIBACTERIAL POVIDONE-IODINE PREPARATIONS": "MISC. DERMATOLOGICAL PRODUCTS",
        "ANGIOTENSIN II RECEPTOR BLOCKERS (ARBS)": "ANGIOTENSIN II RECEPTOR ANTAGONISTS",
        "ATTENTION DEFICIT-HYPERACTIVITY (ADHD) THERAPY, STIMULANT-TYPE": "ATTENTION-DEFICIT/HYPERACTIVITY DISORDER (ADHD) AGENTS",
        "ANTICONVULSANT - MONOSACCHARIDE DERIVATIVES": "ANTICONVULSANTS - MISC.",
        "ANTICONVULSANT - IMINOSTILBENE DERIVATIVES": "ANTICONVULSANTS - MISC.",
        "HERPES ANTIVIRAL AGENT - PURINE ANALOGS": "HERPES AGENTS",
        "ANTIHYPERGLYCEMIC, INCRETIN MIMETIC,GLP-1 RECEPTOR AGONIST ANALOG-TYPE": "INCRETIN MIMETIC AGENTS (GLP-1 RECEPTOR AGONISTS)",
        "PULMONARY FIBROSIS TREATMENT AGENTS - ANTIFIBROTIC THERAPY": "PULMONARY FIBROSIS AGENTS",
        "CMV ANTIVIRAL AGENT - NUCLEOSIDE ANALOGS": "CMV AGENTS",
        "GLUCOCORTICOIDS": "GLUCOCORTICOSTEROIDS",
        "ANTIARRHYTHMIC - CLASS III": "ANTIARRHYTHMICS TYPE III",
        "MU-OPIOID RECEPTOR ANTAGONISTS, PERIPHERALLY-ACTING": "PERIPHERAL OPIOID RECEPTOR ANTAGONISTS",
        "GENERAL ANESTHETIC - PARENTERAL, PHENOL DERIVATIVES": "ANESTHETICS - MISC.",
        "MINERALS AND ELECTROLYTES - PARENTERAL ELECTROLYTE COMBINATIONS": "ELECTROLYTE MIXTURES",
        "LOW MOLECULAR WEIGHT HEPARINS": "HEPARINS AND HEPARINOID-LIKE AGENTS",
        "ANTIEMETIC - CANNABINOID TYPE": "ANTIEMETICS - MISCELLANEOUS",
        "ANTIEMETIC - SELECTIVE SEROTONIN 5-HT3 ANTAGONISTS": "5-HT3 RECEPTOR ANTAGONISTS",
        "MINERALS AND ELECTROLYTES - PARENTERAL ELECTROLYTE COMBINATIONS": "ELECTROLYTE MIXTURES",
        "OXYTOCIC - OXYTOCIN AND ANALOGS": "OXYTOCICS",
        "NSAID ANALGESICS (COX NON-SPECIFIC) - PROPIONIC ACID DERIVATIVES": "NONSTEROIDAL ANTI-INFLAMMATORY AGENTS (NSAIDS)",
        "DERMATOLOGICAL - PROTECTANT COMBINATIONS": "MISC. DERMATOLOGICAL PRODUCTS",
        "HEMOSTATIC SYSTEMIC - ANTIFIBRINOLYTIC AGENTS": "HEMOSTATICS - SYSTEMIC",
        "PULMONARY ANTIHYPERTENSIVE AGENTS-SOLUBLE GUANYLATE CYCLASE STIMULATOR": "PULMONARY HYPERTENSION - SOL GUANYLATE CYCLASE STIMULATOR",
        "BIPOLAR THERAPY AGENTS - ATYPICAL ANTIPSYCHOTICS": "ANTIPSYCHOTICS - MISC.",
        "ANALGESIC OR ANTIPYRETIC NON-OPIOID": "ANALGESICS OTHER",
        "ANTIHISTAMINE - 1ST GENERATION - PIPERIDINES": "ANTIHISTAMINES - PIPERIDINES",
        "LAXATIVE - SALINE AND OSMOTIC": "SALINE LAXATIVES",
        "ANTACID - SIMETHICONE COMBINATIONS": "ANTACID COMBINATIONS",
        "ANALGESIC OPIOID OXYCODONE COMBINATIONS": "ANALGESIC COMBINATIONS",
        "DIURETIC - THIAZIDES AND RELATED": "THIAZIDES AND THIAZIDE-LIKE DIURETICS",
        "NASAL CORTICOSTEROIDS": "NASAL STEROIDS",
        "DERMATOLOGICAL - CALCINEURIN INHIBITORS": "MISC. DERMATOLOGICAL PRODUCTS",
        "HYPERPARATHYROID TREATMENT AGENTS - VITAMIN D ANALOG-TYPE": "ANTITHYROID AGENTS",
        "OPHTHALMIC - DIAGNOSTIC AGENTS": "OPHTHALMICS - MISC.",
        "PLATELET AGGREGATION INHIB-PDESTERASE AND ADENOSINE DEAMINASE INHIBITR": "PLATELET AGGREGATION INHIBITORS",
        "ANTIHISTAMINE - 1ST GENERATION - ETHANOLAMINES": "ANTIHISTAMINES - ETHANOLAMINES",
        "ANTIDEPRESSANT - ALPHA-2 RECEPTOR ANTAGONISTS (NASSA)": "ALPHA-2 RECEPTOR ANTAGONISTS (TETRACYCLICS)",
        "PHARMACEUTICAL ADJUVANT - FLAVORING AGENTS": "PHARMACEUTICAL EXCIPIENTS",
        "ANTICONVULSANT - PYRROLIDINE DERIVATIVES": "ANTICONVULSANTS - MISC.",
        "ANTIPROTOZOAL-ANTIBACTERIAL 1ST GENERATION 2-METHYL-5-NITROIMIDAZOLE": "ANTIPROTOZOAL AGENTS",
        "PULMONARY ARTERIAL HYPERTENSION - SELECTIVE CGMP-PDE5 INHIBITORS": "PULMONARY HYPERTENSION - PHOSPHODIESTERASE INHIBITORS",
        "CMV ANTIVIRAL AGENT - TERMINASE COMPLEX INHIBITORS": "CMV AGENTS",
    }

    # Got these from running the algorithm. Just saving here to save time for future reference
    mapping = {
        "CALCIUM CHANNEL BLOCKERS - DIHYDROPYRIDINES": "CALCIUM CHANNEL BLOCKERS",
        "BETA BLOCKERS CARDIAC SELECTIVE": "BETA BLOCKERS CARDIO-SELECTIVE",
        "INFLAMMATORY BOWEL AGENT - AMINOSALICYLATES AND RELATED AGENTS": "SALICYLATES",
        "DERMATOLOGICAL - TOPICAL LOCAL ANESTHETIC AMIDES": "LOCAL ANESTHETICS - AMIDES",
        "DEXTROSE AND SODIUM CHLORIDE SOLUTIONS": "SODIUM",
        "MINERALS AND ELECTROLYTES - POTASSIUM FOR INJECTION": "POTASSIUM",
        "ANTIFUNGAL - FLUORINATED PYRIMIDINE-TYPE AGENTS": "ANTIFUNGALS",
        "DERMATOLOGICAL - LOCAL ANESTHETIC COMBINATIONS": "LOCAL ANESTHETIC COMBINATIONS",
        "ANTIFUNGAL - GLUCAN SYNTHESIS INHIBITOR, ECHINOCANDINS": "ANTIFUNGAL - GLUCAN SYNTHESIS INHIBITORS",
        "ANALGESIC OPIOID AGONISTS": "OPIOID AGONISTS",
        "PHOSPHATE BINDERS": "PHOSPHATE",
        "ANTIANGINAL - CORONARY VASODILATORS (NITRATES)": "NITRATES",
        "ARTIFICIAL TEARS AND LUBRICANT SINGLE AGENTS": "ARTIFICIAL TEARS AND LUBRICANTS",
        "ANTIANXIETY AGENT - BENZODIAZEPINES": "BENZODIAZEPINES",
        "DERMATOLOGICAL - BURN PRODUCTS ANTI-INFECTIVE": "BURN PRODUCTS",
        "CARDIOVASCULAR SYMPATHOMIMETICS": "SYMPATHOMIMETICS",
        "DIURETIC - LOOP": "LOOP DIURETICS",
        "MINERALS AND ELECTROLYTES - POTASSIUM, ORAL": "POTASSIUM",
        "ANTACID - CALCIUM": "CALCIUM",
        "DIURETIC - CARBONIC ANHYDRASE INHIBITORS": "CARBONIC ANHYDRASE INHIBITORS",
        "METABOLIC MODIFIER - CARNITINE REPLENISHER AGENTS": "METABOLIC MODIFIERS",
        "DIURETIC - SELECTIVE ARGININE VASOPRESSIN V2 RECEPTOR ANTAGONISTS": "VASOPRESSIN RECEPTOR ANTAGONISTS",
        "OPHTHALMIC-INTRAOCULAR PRESSURE REDUCING AGENTS, PROSTAGLANDIN ANALOGS": "PROSTAGLANDINS",
        "OPHTHALMIC ANTIBIOTIC - FLUOROQUINOLONES": "FLUOROQUINOLONES",
        "INSULIN ANALOGS - RAPID ACTING": "INSULIN",
        "MOUTH AND THROAT - ANTISEPTICS": "ANTISEPTICS - MOUTH/THROAT",
        "MINERALS AND ELECTROLYTES - CALCIUM REPLACEMENT": "CALCIUM",
        "DERMATOLOGICAL - TOPICAL LOCAL ANESTHETIC ESTERS": "LOCAL ANESTHETICS - ESTERS",
        "SMOKING DETERRENTS - NICOTINE-TYPE": "SMOKING DETERRENTS",
        "HEPARINS": "HEPARINS AND HEPARINOID-LIKE AGENTS",
        "ABORTIFACIENTS OR CERVICAL RIPENING AGENTS - PROSTAGLANDIN ANALOGS": "PROSTAGLANDINS",
        "URINARY ANTISPASMODIC - SMOOTH MUSCLE RELAXANTS": "ANTISPASMODICS",
        "ANTICOAGULANTS - COUMARIN": "COUMARIN ANTICOAGULANTS",
        "INDIRECT FACTOR XA INHIBITORS": "DIRECT FACTOR XA INHIBITORS",
        "ANTIHISTAMINE - 1ST GENERATION - PHENOTHIAZINES": "PHENOTHIAZINES",
        "ANTIPSYCHOTIC - BUTYROPHENONE DERIVATIVES": "BUTYROPHENONES",
        "INSULINS": "INSULIN",
        "ANTITUBERCULAR - ISONICOTINIC ACID DERIVATIVES": "NICOTINIC ACID DERIVATIVES",
        "ARTIFICIAL TEARS AND LUBRICANT COMBINATIONS": "ARTIFICIAL TEARS AND LUBRICANTS",
        "SYSTEMIC SYMPATHOMIMETIC DECONGESTANTS": "SYMPATHOMIMETIC DECONGESTANTS",
        "MINERALS AND ELECTROLYTES - IRON": "IRON",
        "AMINOGLYCOSIDE ANTIBIOTIC": "AMINOGLYCOSIDES",
        "DMARD - PYRIMIDINE SYNTHESIS INHIBITORS": "PYRIMIDINE SYNTHESIS INHIBITORS",
        "PLATELET AGGREGATION INHIBITORS - PHOSPHODIESTERASE III INHIBITORS": "PLATELET AGGREGATION INHIBITORS",
        "DIAGNOSTIC RADIOPHARMACEUTICALS - RENAL IMAGING": "DIAGNOSTIC RADIOPHARMACEUTICALS",
        "DIAGNOSTIC DRUGS - IN VIVO OTHER": "DIAGNOSTIC DRUGS",
        "ANTIHYPERGLYCEMIC - SULFONYLUREA DERIVATIVES": "SULFONYLUREAS",
        "ANTINEOPLASTIC - ANTIMETABOLITE - FOLIC ACID ANALOGS": "ANTIMETABOLITES",
        "AMINOPENICILLIN ANTIBIOTIC": "AMINOPENICILLINS",
        "OPHTHALMIC - BETA BLOCKERS-CARBONIC ANHYDRASE INHIBITOR COMBINATIONS": "CARBONIC ANHYDRASE INHIBITORS",
        "CARBAPENEM ANTIBIOTICS (THIENAMYCINS)": "CARBAPENEMS",
        "ANTACID - BICARBONATE": "ANTACIDS - BICARBONATE",
        "DERMATOLOGICAL - EMOLLIENT MIXTURES": "EMOLLIENTS",
        "ANTINEOPLASTIC ANTIBIOTIC - ANTHRACYCLINES": "ANTINEOPLASTIC ANTIBIOTICS",
        "PROSTATIC HYPERTROPHY AGENT - ALPHA-1-ADRENOCEPTOR ANTAGONISTS": "PROSTATIC HYPERTROPHY AGENTS",
        "ANTIDIURETIC AND VASOPRESSOR HORMONES": "VASOPRESSORS",
        "LAXATIVE - LUBRICANT": "LUBRICANT LAXATIVES",
        "SODIUM CHLORIDE, PARENTERAL": "SODIUM",
        "THYROID HORMONES - SYNTHETIC T4 (THYROXINE)": "THYROID HORMONES",
        "THYROID HORMONES - SYNTHETIC T3 (TRIIODOTHYRONINE)": "THYROID HORMONES",
        "ANTIPSYCHOTIC - PHENOTHIAZINES, PIPERAZINE": "PHENOTHIAZINES",
        "MINERALS AND ELECTROLYTES - MAGNESIUM": "MAGNESIUM",
        "MUSCULOSKELETAL THERAPY AGENT - VISCOSUPPLEMENTS": "VISCOSUPPLEMENTS",
        "ANTINEOPLASTIC - ANTIMETABOLITE - UREA DERIVATIVES": "ANTIMETABOLITES",
        "TETRACYCLINE ANTIBIOTICS": "TETRACYCLINES",
        "GI ANTISPASMODIC - BELLADONNA ALKALOIDS": "ANTISPASMODICS",
        "ANTIPSYCHOTIC - ATYPICAL DOPAMINE-SEROTONIN ANTAG- BENZISOXAZOLE DERIV": "BENZISOXAZOLES",
        "LINCOSAMIDE ANTIBIOTICS": "LINCOSAMIDES",
        "ASTHMA/COPD -  PHOSPHODIESTERASE-4 (PDE4) INHIBITORS": "PHOSPHODIESTERASE 4 (PDE4) INHIBITORS",
        "GASTRIC ACID SECRETION REDUCING AGENTS - PROTON PUMP INHIBITORS (PPIS)": "PROTON PUMP INHIBITORS",
        "ANTIDIARRHEAL - ANTIPERISTALTIC AGENTS": "ANTIPERISTALTIC AGENTS",
        "B-COMPLEX VITAMIN COMBINATIONS": "B-COMPLEX VITAMINS",
        "ANTIRETROVIRAL-NUCLEOSIDE ANALOGS AND INTEGRASE INHIBITOR COMBINATIONS": "ANTIRETROVIRALS",
        "ANTIEMETIC - PHENOTHIAZINES": "PHENOTHIAZINES",
        "OPIOID ANTITUSSIVE-EXPECTORANT COMBINATIONS": "ANTITUSSIVES",
        "ANTIHYPERLIPIDEMIC - HMG COA REDUCTASE INHIBITORS (STATINS)": "HMG COA REDUCTASE INHIBITORS",
        "ANTIDEPRESSANT - SELECTIVE SEROTONIN REUPTAKE INHIBITORS (SSRIS)": "SELECTIVE SEROTONIN REUPTAKE INHIBITORS (SSRIS)",
        "ANTIHYPERLIPIDEMIC - BILE ACID SEQUESTRANTS": "BILE ACID SEQUESTRANTS",
        "SKELETAL MUSCLE RELAXANT - CENTRAL MUSCLE RELAXANTS": "CENTRAL MUSCLE RELAXANTS",
        "SEDATIVE-HYPNOTIC - BENZODIAZEPINES": "BENZODIAZEPINES",
        "ANTIRETROVIRAL - NON-NUCLEOSIDE REVERSE TRANSCRIPTASE INHIB (NNRTI)": "ANTIRETROVIRALS",
        "MULTIVITAMIN AND MINERAL COMBINATIONS": "MINERAL COMBINATIONS",
        "GLYCOPEPTIDE ANTIBIOTICS": "GLYCOPEPTIDES",
        "DERMATOLOGICAL - EMOLLIENTS": "EMOLLIENTS",
        "MIGRAINE THERAPY - SELECTIVE SEROTONIN AGONISTS 5-HT(1)": "SEROTONIN AGONISTS",
        "MINERALS AND ELECTROLYTES - BICARBONATE PRODUCING OR CONTAINING AGENTS": "BICARBONATES",
        "LOCAL ANESTHETIC - AMIDES": "LOCAL ANESTHETICS - AMIDES",
        "AMINOPENICILLIN ANTIBIOTIC - BETA-LACTAMASE INHIBITOR COMBINATIONS": "AMINOPENICILLINS",
        "URINARY ANTISPASMODIC - ANTICHOL., M(3) MUSCARINIC SELECTIVE (BLADDER)": "ANTISPASMODICS",
        "DERMATOLOGICAL - ANTIFUNGAL-GLUCOCORTICOID COMBINATIONS": "ANTIFUNGALS",
        "PULMONARY ARTERIAL HYPERTENSION - ENDOTHELIN RECEPTOR ANTAGONISTS": "PULMONARY HYPERTENSION - ENDOTHELIN RECEPTOR ANTAGONISTS",
        "MINERALS AND ELECTROLYTES - ZINC": "ZINC",
        "INSULIN ANALOGS - LONG ACTING": "INSULIN",
        "LOCAL ANESTHETIC - SYMPATHOMIMETIC COMBINATIONS": "SYMPATHOMIMETICS",
        "ANTIRETROVIRAL-NUCLEOSIDE, NUCLEOTIDE ANALOGS AND NON-NUCLEOSIDE RTI": "ANTIRETROVIRALS",
        "OPHTHALMIC - CARBONIC ANHYDRASE INHIBITORS": "CARBONIC ANHYDRASE INHIBITORS",
        "ANTIFUNGAL - AMPHOTERIC POLYENE MACROLIDES": "ANTIFUNGALS",
        "OPHTHALMIC - LOCAL ANESTHETIC ESTERS": "OPHTHALMIC LOCAL ANESTHETICS",
        "HUMAN INSULINS - SHORT ACTING": "INSULIN",
        "ANTIHYPERGLYCEMIC - MEGLITINIDE ANALOGS": "MEGLITINIDE ANALOGUES",
        "DIRECT ACTING VASODILATORS": "VASODILATORS",
        "LAXATIVE - STIMULANT": "STIMULANT LAXATIVES",
        "SALICYLATE ANALGESICS": "SALICYLATES",
        "DIAGNOSTIC DRUGS - CARDIOVASCULAR": "DIAGNOSTIC DRUGS",
    }

    missing_mapping = []

    for i, reference in enumerate(only_cedars):

        comparators = []
        sorted_results = []
        partial_results = []
        single_results = []
        set_results = []

        if reference in mapping.keys():
            print("Already have mapping: ", reference, " -> ", mapping[reference])
            print()
            continue

        if reference in manual_mapping.keys():
            print(
                "Overrided with manual mapping: ",
                reference,
                " -> ",
                manual_mapping[reference],
            )
            print()
            continue

        for comparator in all_ucla:
            comparators.append(comparator)
            sorted_results.append(fuzz.token_sort_ratio(reference, comparator))
            partial_results.append(fuzz.partial_ratio(reference, comparator))
            single_results.append(fuzz.ratio(reference, comparator))
            set_results.append(fuzz.partial_token_sort_ratio(reference, comparator))

        sorted_distances = (
            DataFrame({"comparator": comparators, "similarity": sorted_results})
            .sort_values(by=["similarity"], ascending=False)
            .reset_index()
        )
        partial_distances = (
            DataFrame({"comparator": comparators, "similarity": partial_results})
            .sort_values(by=["similarity"], ascending=False)
            .reset_index()
        )
        single_distances = (
            DataFrame({"comparator": comparators, "similarity": single_results})
            .sort_values(by=["similarity"], ascending=False)
            .reset_index()
        )

        if reference in exceptions:
            print("Exception")
            print(
                "Top candidates: ",
                "\n",
                sorted_distances["comparator"][0],
                sorted_distances["similarity"][0],
                "\n",
                partial_distances["comparator"][0],
                partial_distances["similarity"][0],
                "\n",
                single_distances["comparator"][0],
                single_distances["similarity"][0],
            )
            missing_mapping.append(reference)
            continue

        print(f"Comparing {reference}")
        if sorted_distances["similarity"][0] >= 90:
            print(
                "Found similar: ",
                sorted_distances["comparator"][0],
                sorted_distances["similarity"][0],
            )
            mapping[reference] = sorted_distances["comparator"][0]
        elif single_distances["similarity"][0] >= 90:
            diff = difflib.ndiff(reference, single_distances["comparator"][0])

            if not any(c.isnumeric() for c in diff):
                if not any(c in ["I", "N"] for c in diff):
                    mapping[reference] = single_distances["comparator"][0]
                    print(
                        "Found similar: ",
                        single_distances["comparator"][0],
                        single_distances["similarity"][0],
                    )
            else:
                missing_mapping.append(reference)
                print(
                    "Reject: ",
                    single_distances["comparator"][0],
                    single_distances["similarity"][0],
                )
        elif partial_distances["similarity"][0] >= 90:
            diff = difflib.ndiff(reference, partial_distances["comparator"][0])

            if not any(c.isnumeric() for c in diff):
                mapping[reference] = partial_distances["comparator"][0]
                print(
                    "Found similar: ",
                    partial_distances["comparator"][0],
                    partial_distances["similarity"][0],
                )
        else:
            print("No similarity")
            print(
                "Top candidates: ",
                "\n",
                sorted_distances["comparator"][0],
                sorted_distances["similarity"][0],
                "\n",
                partial_distances["comparator"][0],
                partial_distances["similarity"][0],
                "\n",
                single_distances["comparator"][0],
                single_distances["similarity"][0],
            )
            missing_mapping.append(reference)

        print()

    print("Manually mapped: ", len(manual_mapping))
    print("Automatically mapped: ", len(mapping))
    print("Missing: ", len(missing_mapping))

    mapping = mapping | manual_mapping
    with open(join(args.cedars_crrt_data_dir, "Medications_Mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)


def main():
    load_cli_args()
    args = init_cli_args()

    cedars_meds = read_files_and_combine([FILE_NAMES["rx"]], args.cedars_crrt_data_dir)
    ucla_meds = read_files_and_combine([FILE_NAMES["rx"]], args.ucla_crrt_data_dir)

    cedars_meds = cedars_meds.dropna(subset=["PHARM_SUBCLASS"])
    ucla_meds = ucla_meds.dropna(subset=["PHARM_SUBCLASS"])

    ucla_meds["PHARM_SUBCLASS"] = ucla_meds["PHARM_SUBCLASS"].str.upper()
    ucla_meds = ucla_meds[
        ~ucla_meds["PHARM_SUBCLASS"].str.contains("EACH")
    ]  # 277 cases
    ucla_meds = ucla_meds[~ucla_meds["PHARM_SUBCLASS"].str.isnumeric()]  # 251 cases

    cedars_meds["PHARM_SUBCLASS"] = cedars_meds["PHARM_SUBCLASS"].str.upper()
    cedars_meds = cedars_meds[~cedars_meds["PHARM_SUBCLASS"].str.contains("EACH")]
    cedars_meds = cedars_meds[~cedars_meds["PHARM_SUBCLASS"].str.isnumeric()]

    only_cedars = (set(cedars_meds["PHARM_SUBCLASS"].unique())).difference(
        set(ucla_meds["PHARM_SUBCLASS"].unique())
    )
    all_ucla = set(ucla_meds["PHARM_SUBCLASS"].unique())

    create_and_serialize_medication_mapping_dict(args, only_cedars, all_ucla)


if __name__ == "__main__":
    main()
