"""يضبط مجلد بيانات مؤقتاً قبل استيراد أي وحدة حتى لا تلوّث الاختبارات data/study.db."""
import os
import tempfile

os.environ["BAC_DATA_DIR"] = tempfile.mkdtemp(prefix="bac_tests_")
