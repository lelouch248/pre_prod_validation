import os
import snowflake.connector
from dotenv import load_dotenv
from difflib import unified_diff
import pandas as pd
import re

load_dotenv()

#############################
# CONFIG
#############################

SUPPORTED_OBJECTS = ["TABLE", "VIEW"]

SCHEMA_LIST = [
    "CORE",
    "ANALYTICS",
    "PRESENTATION"
]


#############################
# SNOWFLAKE CONNECTION
#############################

def get_connection(env_prefix, schema):
    """
    Create Snowflake connection via EXTERNAL BROWSER authentication.
    Env prefix examples: PRD, TST, STG
    """
    return snowflake.connector.connect(
        user=os.getenv("USER"),
        account=os.getenv("ACCOUNT"),
        authenticator="EXTERNALBROWSER",
        warehouse=os.getenv(f"{env_prefix}_WAREHOUSE"),
        database=os.getenv(f"{env_prefix}_DATABASE"),
        schema=schema,
        role=os.getenv(f"{env_prefix}_ROLE")
    )


#############################
# DDL NORMALIZATION
#############################

def normalize_ddl(ddl):
    """
    Improve DDL comparison accuracy:
    - Remove comments
    - Collapse whitespace
    - Lowercase
    - Replace env tags
    - Normalize brackets, commas, spacing
    """

    ddl = re.sub(r'--.*', '', ddl)                           # remove inline comments
    ddl = re.sub(r'/\*.*?\*/', '', ddl, flags=re.DOTALL)     # remove block comments
    ddl = re.sub(r'_TST_|_PRD_|_DEV_', '_ENV_', ddl)         # environment replace
    ddl = re.sub(r"\s*\(\s*", "(", ddl)
    ddl = re.sub(r"\s*\)\s*", ")", ddl)
    ddl = re.sub(r"\s*,\s*", ",", ddl)
    ddl = re.sub(r'\s+', ' ', ddl)
    ddl = ddl.strip().lower()

    # reorder column lines inside CREATE TABLE for cleaner comparison
    if ddl.startswith("create table"):
        try:
            inside = ddl[ddl.index("(")+1:ddl.rindex(")")]
            cols = [c.strip() for c in inside.split(",")]
            cols = sorted(cols)
            ddl = ddl[:ddl.index("(")+1] + ",".join(cols) + ")"
        except:
            pass

    return ddl


#############################
# FETCH OBJECT METADATA
#############################

def fetch_objects(conn, env_prefix):
    cursor = conn.cursor()
    objects = {}

    inclusion_list = [
        f"SDM_{env_prefix}_ONECUSTOMER_DB",
        f"SRC_{env_prefix}_DATAEDITOR_DB"
    ]

    for obj_type in SUPPORTED_OBJECTS:
        cursor.execute(f"SHOW {obj_type}s")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        df = pd.DataFrame(rows, columns=columns)
        df = df[df['database_name'].isin(inclusion_list)]  # filter dbs

        for index, row in df.iterrows():
            name = row['name']
            full_key = f"{obj_type}.{name}"
            try:
                cursor.execute(f"SELECT GET_DDL('{obj_type}', '{name}')")
                ddl = cursor.fetchone()[0]
            except:
                ddl = "-- ERROR FETCHING DDL --"

            ddl = normalize_ddl(ddl)
            objects[full_key] = ddl

        print(f"Fetched {len(df)} {obj_type}s from {env_prefix}")

    return objects


#############################
# DIFF SUMMARY
#############################

def summarize_change(diff_text):
    additions = diff_text.count("+ ")
    deletions = diff_text.count("- ")

    if additions and deletions:
        return f"Modified ({additions} additions, {deletions} deletions)"
    elif additions:
        return f"Added ({additions} new lines)"
    elif deletions:
        return f"Removed ({deletions} lines removed)"
    return "Formatting-only changes"


#############################
# COMPARISON
#############################

def compare_objects(prod_objects, tst_objects):
    missing_in_prod = []
    changed_objects = []

    for obj, tst_ddl in tst_objects.items():
        prod_ddl = prod_objects.get(obj)

        if prod_ddl is None:
            missing_in_prod.append(obj)
            continue

        if prod_ddl != tst_ddl:
            diff = "\n".join(
                unified_diff(
                    prod_ddl.splitlines(),
                    tst_ddl.splitlines(),
                    fromfile="PROD",
                    tofile="TST",
                    lineterm=""
                )
            )

            if diff.strip():
                changed_objects.append((obj, diff, summarize_change(diff)))

    return missing_in_prod, changed_objects


#############################
# MARKDOWN REPORT GENERATION
#############################

def generate_md_report(missing_dict, changed_dict, target_env_prefix):
    filename = f"Snowflake_Comparison_Report_{target_env_prefix}.md"

    with open(filename, "w") as f:
        f.write(f"# Snowflake Comparison Report: {target_env_prefix} → PROD\n\n")

        for schema in missing_dict.keys():
            f.write(f"## 📂 Schema: {schema}\n\n")

            f.write("### ❌ Objects missing in PROD\n")
            if not missing_dict[schema]:
                f.write("- None\n")
            else:
                for obj in missing_dict[schema]:
                    f.write(f"- {obj}\n")

            f.write("\n### ⚠️ Objects with DDL differences\n")
            if not changed_dict[schema]:
                f.write("- None\n\n")
            else:
                for obj, diff, summary in changed_dict[schema]:
                    f.write(f"#### {obj}\n")
                    f.write(f"- Summary: **{summary}**\n")
                    f.write("```diff\n")
                    f.write(diff + "\n")
                    f.write("```\n")

        f.write("\n---\nReport Generated Automatically\n")

    print(f"\n📄 Markdown report created: {filename}")


#############################
# MAIN EXECUTION
#############################

def main(target_env_prefix="TST"):
    results_missing = {}
    results_changes = {}

    print("🔐 Authenticating using External Browser...")

    for schema in SCHEMA_LIST:
        print(f"\n=============== SCHEMA: {schema} =================")

        prod_conn = get_connection("PRD", schema)
        tgt_conn = get_connection(target_env_prefix, schema)

        print("Fetching PROD metadata...")
        prod_objects = fetch_objects(prod_conn, "PRD")

        print(f"Fetching {target_env_prefix} metadata...")
        tgt_objects = fetch_objects(tgt_conn, target_env_prefix)

        print("Comparing differences...")
        missing, changed = compare_objects(prod_objects, tgt_objects)

        results_missing[schema] = missing
        results_changes[schema] = changed

        prod_conn.close()
        tgt_conn.close()

    generate_md_report(results_missing, results_changes, target_env_prefix)
    print("\n🏁 Comparison Completed.\n")


if __name__ == "__main__":
    main("TST")  # modify to STG / DEV / QA