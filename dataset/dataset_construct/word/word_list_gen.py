CONFERENCE = [
    # NLP
    "ACL", "NAACL", "EMNLP",
    # CV
    "CVPR", "ICCV", "ECCV",
    # Speech
    "INTERSPEECH", "ICASSP",
    # AI
    "IJCAI", "AAAI",
    # ML
    "ICLR", "ICML", "NeurIPS",
    # Data Mining
    "KDD", "WSDM", "WWW",
    # Database
    "SIGMOD", "VLDB", "ICDE",
    # IR
    "SIGIR",
    # HCI
    "CHI"
]
YEAR = ["2020", "2021", "2022", "2023"]
FORM = ["paper", "talk", "workshop"]

with open("conf_year_form.txt", "w") as f:
    for conf in CONFERENCE:
        for year in YEAR:
            for form in FORM:
                f.write(f"{conf} {year} {form}\n")